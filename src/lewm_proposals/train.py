from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import time

import numpy as np
import torch

from .common import (
    append_jsonl,
    atomic_torch_save,
    read_json,
    resolve_device,
    runtime_metadata,
    seed_all,
    synchronize,
    write_json,
)
from .data import WindowStore, validate_manifest
from .flow import JointFlow


def precision_context(device, precision):
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "fp32":
        return nullcontext
    if precision != "bf16" or device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Use fp32, or bf16 on a supported CUDA GPU")
    return lambda: torch.autocast("cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def validate(model, store, settings, device, amp):
    model.eval()
    rng = np.random.default_rng(731)
    noise = torch.Generator().manual_seed(731)
    losses = []
    for _ in range(settings["val_batches"]):
        batch = store.sample(settings["batch_size"], rng, device)
        with amp():
            loss, _ = model.loss(
                batch, noise, settings["state_loss_weight"], settings["action_loss_weight"]
            )
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def train(cfg, output, *, resume=False, stop_after=None):
    settings = cfg["train"]
    seed_all(settings["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = resolve_device(cfg["device"])
    amp = precision_context(device, settings["precision"])
    manifest = read_json(cfg["manifest"])
    validate_manifest(manifest)
    training = WindowStore(cfg["cache"], "train")
    validation = WindowStore(cfg["cache"], "val")
    if training.meta["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Cache does not belong to this split manifest")
    model = JointFlow(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(settings["seed"])
    noise = torch.Generator().manual_seed(settings["seed"] + 1)
    root = Path(output)
    step, best, elapsed = 0, float("inf"), 0.0
    if resume:
        state = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        if state["cache_id"] != training.cache_id or state["model_config"] != cfg["model"]:
            raise ValueError("Resume cache/model mismatch")
        if state["train_config"] != settings:
            raise ValueError(
                "Exact resume requires unchanged train config; start a new run for new settings"
            )
        if state["device_type"] != device.type:
            raise ValueError("Exact resume requires the same device type")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer state tensors must follow the parameters after CPU checkpoint loading.
        for opt in optimizer.state.values():
            for key, value in opt.items():
                if torch.is_tensor(value) and key != "step":
                    opt[key] = value.to(device)
        rng.bit_generator.state = state["numpy_rng"]
        noise.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, best, elapsed = state["step"], state["best_val"], state["elapsed_seconds"]
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "config.json", cfg)
        write_json(root / "runtime.json", runtime_metadata())
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    stop = min(settings["steps"], step + stop_after) if stop_after else settings["steps"]
    write_json(root / "status.json", {"state": "running", "step": step}, replace=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {parameters:,}; device={device}; valid windows={training.cumulative[-1]}",
        flush=True,
    )

    def checkpoint():
        return {
            "schema": "joint-flow-checkpoint-v1",
            "model": model.state_dict(),
            "model_config": cfg["model"],
            "train_config": settings,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "cache_id": training.cache_id,
            "cache_metadata": training.meta,
            "latent_stats": training.latent_stats,
            "action_stats": training.action_stats,
            "best_val": best,
            "numpy_rng": rng.bit_generator.state,
            "noise_rng": noise.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "device_type": device.type,
            "elapsed_seconds": elapsed + time.perf_counter() - started,
        }

    try:
        model.train()
        while step < stop:
            warmup = settings["warmup_steps"]
            if step < warmup:
                factor = (step + 1) / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, settings["steps"] - warmup)
                factor = 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = settings["learning_rate"] * factor
            batch = training.sample(settings["batch_size"], rng, device)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss, parts = model.loss(
                    batch, noise, settings["state_loss_weight"], settings["action_loss_weight"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if step % settings["log_every"] == 0 or step == stop:
                synchronize(device)
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    **{k: float(v.detach()) for k, v in parts.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed + time.perf_counter() - started,
                }
                append_jsonl(root / "train.jsonl", row)
                print(row, flush=True)
            if step % settings["val_every"] == 0 or step == stop:
                value = validate(model, validation, settings, device, amp)
                improved = value < best
                if improved:
                    best = value
                state = checkpoint()
                atomic_torch_save(state, root / "last.pt")
                if improved:
                    atomic_torch_save(state, root / "best.pt")
                append_jsonl(root / "validation.jsonl", {"step": step, "loss": value})
        synchronize(device)
        write_json(
            root / "status.json",
            {
                "state": "complete" if step == settings["steps"] else "paused",
                "step": step,
                "parameters": parameters,
                "best_val": best,
                "elapsed_seconds": elapsed + time.perf_counter() - started,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            },
            replace=True,
        )
    except BaseException as exc:
        write_json(
            root / "status.json",
            {"state": "failed", "step": step, "error": f"{type(exc).__name__}: {exc}"},
            replace=True,
        )
        raise
    return root


def load_generator(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("schema") != "joint-flow-checkpoint-v1":
        raise ValueError("Unsupported generator checkpoint")
    model = JointFlow(state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, state
