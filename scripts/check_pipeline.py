"""Opt-in engineering integration check with real LeWM weights and a tiny fresh Cube fixture.

Runs the full entry-point chain using three short scripted trajectories and only two
generator updates. This checks wiring, not learned control quality or the research claim.
No downloaded expert dataset is required. Inputs and outputs are explicitly selected.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--render-backend", default="egl")
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MUJOCO_GL"] = args.render_backend
    from lewm_proposals.vendor.cube_env import CubeEnv
    from lewm_proposals.evaluate import close_env

    arrays = {
        k: []
        for k in (
            "pixels",
            "action",
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        )
    }
    lengths = []
    for seed in range(3):
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
            for t in range(30):
                action = np.array(
                    [0.02 * np.sin(t / 5 + seed), 0.02 * np.cos(t / 5), 0, 0, 0], dtype=np.float32
                )
                arrays["pixels"].append(np.asarray(env.render()).copy())
                arrays["action"].append(action)
                arrays["qpos"].append(env._data.qpos.copy())
                arrays["qvel"].append(env._data.qvel.copy())
                obj = env._data.joint("object_joint_0").qpos.copy()
                arrays["privileged_block_0_pos"].append(obj[:3])
                arrays["privileged_block_0_quat"].append(obj[3:])
                env.step(action)
            lengths.append(30)
        finally:
            close_env(env)
    with h5py.File(out / "fixture.h5", "w") as f:
        for key, values in arrays.items():
            f.create_dataset(key, data=np.asarray(values), compression="gzip")
        f["ep_len"] = lengths
        f["ep_offset"] = np.arange(3) * 30
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/cube.yaml").read_text(encoding="utf-8")
    )
    cfg.update(
        dataset=str(out / "fixture.h5"),
        weights=str(Path(args.weights).resolve()),
        manifest=str(out / "split.json"),
        cache=str(out / "latents.h5"),
        device="cpu",
        cpu_threads=2,
    )
    cfg["split"].update(train_episodes=1, val_episodes=1, test_episodes=1)
    cfg["train"].update(
        steps=2,
        batch_size=2,
        val_every=1,
        val_batches=1,
        log_every=1,
        warmup_steps=1,
        precision="fp32",
    )
    cfg["eval"].update(
        cases=1,
        planner_seeds=[11],
        candidates=2,
        flow_steps=2,
        cem_samples=4,
        cem_iterations=1,
        cem_elites=2,
        render_backend=args.render_backend,
    )
    config = out / "check.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    commands = [
        ["doctor", "--load-model"],
        ["inspect"],
        ["split"],
        ["encode"],
        ["train", "--output", str(out / "training")],
        [
            "evaluate",
            "--checkpoint",
            str(out / "training/best.pt"),
            "--output",
            str(out / "evaluation"),
        ],
    ]
    for command in commands:
        subprocess.run(
            [sys.executable, "-m", "lewm_proposals", *command, "--config", str(config)], check=True
        )
    from lewm_proposals.common import read_json

    status = read_json(out / "evaluation/status.json")
    if status["state"] != "complete" or status["completed"] != 3:
        raise RuntimeError("Integration check did not complete all three methods")
    print(
        "ENGINEERING CHECK COMPLETE: three methods × 50 steps; this is not benchmark evidence.",
        flush=True,
    )


if __name__ == "__main__":
    main()
