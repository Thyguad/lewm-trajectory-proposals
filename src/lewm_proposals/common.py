from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    m, e, t = cfg["model"], cfg["eval"], cfg["train"]
    if (m["latent_dim"], m["action_dim"], m["action_block"], m["horizon"]) != (192, 5, 5, 5):
        raise ValueError("Stage one fixes Cube latent/action/block/horizon to 192/5/5/5")
    if e["execute_steps"] != 5 or e["total_steps"] != 50:
        raise ValueError("Stage-one evaluation fixes execute_steps=5, total_steps=50")
    if m["width"] % m["heads"] or m["time_dim"] % 2:
        raise ValueError("Invalid attention width or time embedding dimension")
    if not 0 < e["cem_elites"] <= e["cem_samples"]:
        raise ValueError("Invalid CEM elite count")
    if (
        min(
            t["steps"],
            t["batch_size"],
            t["val_every"],
            t["val_batches"],
            t["log_every"],
            e["candidates"],
            e["flow_steps"],
            e["cases"],
            e["cem_iterations"],
            e["score_batch_size"],
        )
        <= 0
    ):
        raise ValueError("Counts must be positive")
    if e["split"] not in {"val", "test"}:
        raise ValueError("Evaluation must use val or test")
    if not e["methods"] or not set(e["methods"]) <= {"cem", "flow_direct", "flow_ranked"}:
        raise ValueError("Unknown evaluation method")
    if len(set(e["methods"])) != len(e["methods"]):
        raise ValueError("Duplicate evaluation methods")
    return cfg


def resolve_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derived_seed(seed, label):
    return (int(seed) + int(hashlib.sha256(str(label).encode()).hexdigest()[:8], 16)) % (2**31)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def runtime_metadata():
    packages = {}
    for name in (
        "torch",
        "numpy",
        "h5py",
        "hdf5plugin",
        "transformers",
        "mujoco",
        "dm-control",
        "ogbench",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": dirty,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)
