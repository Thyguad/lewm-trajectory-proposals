"""Evaluator owns simulator state and success truth; planner receives only visual latents."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import (
    append_jsonl,
    derived_seed,
    digest,
    file_hash,
    read_json,
    resolve_device,
    runtime_metadata,
    seed_all,
    synchronize,
    write_json,
)
from .data import CubeData, evaluation_cases, validate_manifest
from .frozen import FrozenLeWM
from .planning import Planner
from .train import load_generator


def make_env(settings, initial, target, seed):
    backend = settings["render_backend"]
    if backend != "auto":
        os.environ["MUJOCO_GL"] = backend
    from .vendor.cube_env import CubeEnv

    env = CubeEnv(
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )
    try:
        env.reset(seed=seed, options={"variation": []})
        env.set_state(np.asarray(initial["qpos"]), np.asarray(initial["qvel"]))
        env.set_target_pos(0, target["pos"], target["quat"])
        env.pre_step()
        env.post_step()
        if (
            env.action_space.shape != (5,)
            or not np.all(env.action_space.low == -1)
            or not np.all(env.action_space.high == 1)
        ):
            raise ValueError("Unexpected Cube action space")
        return env
    except BaseException:
        close_env(env)
        raise


def close_env(env):
    renderer = getattr(env, "_renderer", None)
    if renderer is not None:
        renderer.close()
    env.close()


def object_distance(env, target):
    # Evaluation only: never passed into Planner, G, encoder or predictor.
    pos = env._data.joint("object_joint_0").qpos[:3].copy()
    return float(np.linalg.norm(pos - target["pos"]))


def run_case(env, target, goal_image, frozen, planner, settings, source_image):
    synchronize(frozen.device)
    started = time.perf_counter()
    goal = frozen.encode(goal_image[None])[0]
    frame = np.asarray(env.render())
    initial_hash = hashlib.sha256(frame.tobytes()).hexdigest()
    initial_error = float(np.abs(frame.astype(np.float32) - source_image.astype(np.float32)).mean())
    distance = object_distance(env, target)
    initial_success = distance <= settings["success_distance_m"]
    ever_success = initial_success
    plans, history = [], []
    steps = 0
    while steps < settings["total_steps"]:
        synchronize(frozen.device)
        decision_started = time.perf_counter()
        current = frozen.encode(frame[None])[0]
        actions, info = planner.plan(current, goal)
        synchronize(frozen.device)
        info["decision_seconds"] = time.perf_counter() - decision_started
        info["at_step"] = steps
        plans.append(info)
        for action in actions[: min(settings["execute_steps"], settings["total_steps"] - steps)]:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if truncated or terminated:
                raise RuntimeError(
                    "Unexpected environment termination; protocol requires 50 actual steps"
                )
            distance = object_distance(env, target)
            ever_success |= distance <= settings["success_distance_m"]
            history.append({"step": steps, "action": action.tolist(), "distance_m": distance})
        frame = np.asarray(env.render())
    synchronize(frozen.device)
    return {
        "endpoint_success": distance <= settings["success_distance_m"],
        "ever_success": bool(ever_success),
        "initial_success": bool(initial_success),
        "final_distance_m": distance,
        "steps": steps,
        "plans": plans,
        "trajectory": history,
        "initial_frame_sha256": initial_hash,
        "initial_render_mae_vs_dataset": initial_error,
        "episode_wall_seconds": time.perf_counter() - started,
    }


def evaluate(cfg, output, checkpoint=None, *, resume=False):
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = resolve_device(cfg["device"])
    settings = cfg["eval"]
    manifest = read_json(cfg["manifest"])
    validate_manifest(manifest)
    frozen = FrozenLeWM(
        cfg["weights"], device, manifest["action_stats"], settings["score_batch_size"]
    )
    model, ckpt = None, None
    if any(m != "cem" for m in settings["methods"]):
        if checkpoint is None:
            raise ValueError("Supply --checkpoint for flow methods")
        model, ckpt = load_generator(checkpoint, device)
        if ckpt["cache_metadata"]["manifest_id"] != manifest["manifest_id"]:
            raise ValueError("Generator was trained on a different split")
        if ckpt["cache_metadata"]["encoder"] != frozen.identity:
            raise ValueError("Frozen checkpoint differs from the generator's feature encoder")
        if ckpt["action_stats"] != manifest["action_stats"] or ckpt["model_config"] != cfg["model"]:
            raise ValueError("Generator normalization or architecture mismatch")
    root = Path(output)
    contract = {
        "config": cfg,
        "manifest_id": manifest["manifest_id"],
        "frozen": frozen.identity,
        "generator_sha256": file_hash(checkpoint) if checkpoint else None,
        "environment_source_sha256": file_hash(Path(__file__).parent / "vendor/cube_env.py"),
    }
    contract_id = digest(contract)
    if resume:
        if read_json(root / "contract.json")["contract_id"] != contract_id:
            raise ValueError("Cannot resume evaluation with changed inputs/settings")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "contract.json", {**contract, "contract_id": contract_id})
        write_json(root / "runtime.json", runtime_metadata())
    done, frame_hashes = set(), {}
    results_path = root / "episodes.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["planner_seed"], row["method"])
            if key in done:
                raise ValueError("Duplicate evaluation records")
            done.add(key)
            frame_hashes[row["case_id"]] = row["initial_frame_sha256"]
    expected_count = settings["cases"] * len(settings["planner_seeds"]) * len(settings["methods"])
    write_json(
        root / "status.json",
        {"state": "running", "completed": len(done), "expected": expected_count},
        replace=True,
    )
    try:
        with CubeData(cfg["dataset"]) as source:
            if source.identity() != manifest["source"]:
                raise ValueError("Evaluation dataset differs from the split manifest")
            cases = evaluation_cases(
                source, manifest, settings["split"], settings["cases"], settings["case_seed"]
            )
            if not resume:
                write_json(root / "cases.json", cases)
            elif read_json(root / "cases.json") != cases:
                raise ValueError("Evaluation cases changed")
            for case in cases:
                initial, target, goal, source_image = source.initial_and_goal(case)
                env_seed = derived_seed(settings["case_seed"], case["case_id"])
                for seed in settings["planner_seeds"]:
                    for method in settings["methods"]:
                        key = (case["case_id"], seed, method)
                        if key in done:
                            continue
                        # Recreate identically for each method: no method-dependent reset branches.
                        seed_all(env_seed)
                        env = make_env(settings, initial, target, env_seed)
                        try:
                            planner = Planner(
                                method,
                                frozen,
                                settings,
                                derived_seed(seed, case["case_id"]),
                                model,
                                ckpt,
                            )
                            result = run_case(
                                env, target, goal, frozen, planner, settings, source_image
                            )
                        finally:
                            close_env(env)
                        old_hash = frame_hashes.setdefault(
                            case["case_id"], result["initial_frame_sha256"]
                        )
                        if old_hash != result["initial_frame_sha256"]:
                            raise RuntimeError(
                                "Paired methods did not receive identical initial pixels"
                            )
                        append_jsonl(
                            results_path,
                            {
                                "contract_id": contract_id,
                                **case,
                                "planner_seed": seed,
                                "method": method,
                                **result,
                            },
                        )
                        done.add(key)
                        print(
                            f"{len(done)}/{expected_count} {case['case_id']} {method} seed={seed} "
                            f"success={result['endpoint_success']} distance={result['final_distance_m']:.4f}m",
                            flush=True,
                        )
        write_json(
            root / "status.json",
            {"state": "complete", "completed": len(done), "expected": expected_count},
            replace=True,
        )
    except BaseException as exc:
        write_json(
            root / "status.json",
            {
                "state": "failed",
                "completed": len(done),
                "expected": expected_count,
                "error": f"{type(exc).__name__}: {exc}",
            },
            replace=True,
        )
        raise
    summarize(root)


def summarize(root):
    root = Path(root)
    status = read_json(root / "status.json")
    if status["state"] != "complete":
        raise ValueError("Evaluation is incomplete; finish or resume before summarizing")
    rows = [
        json.loads(s) for s in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != status["expected"]:
        raise ValueError("Result count does not match the completed protocol")
    groups = {}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        plans = [p for r in selected for p in r["plans"]]
        groups[method] = {
            "episodes": len(selected),
            "endpoint_success_rate": float(np.mean([r["endpoint_success"] for r in selected])),
            "ever_success_rate": float(np.mean([r["ever_success"] for r in selected])),
            "initial_success_rate": float(np.mean([r["initial_success"] for r in selected])),
            "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in selected])),
            "mean_decision_seconds": float(np.mean([p["decision_seconds"] for p in plans])),
            "p95_decision_seconds": float(
                np.percentile([p["decision_seconds"] for p in plans], 95)
            ),
            "predictor_transitions": int(sum(p["predictor_transitions"] for p in plans)),
            "flow_forward_calls": int(sum(p.get("flow_forward_calls", 0) for p in plans)),
            "success_by_planner_seed": {
                str(s): float(
                    np.mean([r["endpoint_success"] for r in selected if r["planner_seed"] == s])
                )
                for s in sorted({r["planner_seed"] for r in selected})
            },
        }
    paired = {}
    index = {(r["case_id"], r["planner_seed"], r["method"]): r for r in rows}
    if "cem" in groups:
        for method in groups.keys() - {"cem"}:
            by_case = {}
            for r in rows:
                if r["method"] == method:
                    baseline = index[(r["case_id"], r["planner_seed"], "cem")]
                    by_case.setdefault(r["case_id"], []).append(
                        int(r["endpoint_success"]) - int(baseline["endpoint_success"])
                    )
            values = np.array([np.mean(v) for v in by_case.values()])
            rng = np.random.default_rng(512)
            # Cluster by source episode/case, not by correlated planner seeds.
            bootstrap = np.array(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
            )
            paired[method] = {
                "success_delta_vs_cem": float(values.mean()),
                "case_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            }
    summary = {
        "metrics": groups,
        "paired": paired,
        "scope": "Local Cube 25-step future goals, 50 control steps; position only. Planner seeds are not training seeds.",
    }
    write_json(root / "summary.json", summary, replace=True)
    lines = [
        "# Cube 第一阶段结果",
        "",
        summary["scope"],
        "",
        "| 方法 | 终点成功率 | 曾经达标率 | 终点距离 m | 平均决策秒 | P95 决策秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, r in groups.items():
        lines.append(
            f"| {method} | {r['endpoint_success_rate']:.3f} | {r['ever_success_rate']:.3f} | "
            f"{r['mean_final_distance_m']:.4f} | {r['mean_decision_seconds']:.4f} | {r['p95_decision_seconds']:.4f} |"
        )
    lines += [
        "",
        "规划延迟包含当前图像编码、候选生成和筛选；不含环境执行。初始化、目标编码和渲染计入逐回合总耗时。",
        "本报告由真实运行日志生成，不推断未运行的训练种子或跨环境收益。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
