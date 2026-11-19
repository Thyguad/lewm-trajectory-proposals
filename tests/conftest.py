import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from lewm_proposals.common import digest
from lewm_proposals.data import make_split


@pytest.fixture
def small_config(tmp_path):
    torch.set_num_threads(1)
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/cube.yaml").read_text())
    cfg.update(
        dataset=str(tmp_path / "source.h5"),
        manifest=str(tmp_path / "split.json"),
        cache=str(tmp_path / "latents.h5"),
        device="cpu",
        cpu_threads=1,
    )
    cfg["split"].update(train_episodes=2, val_episodes=1, test_episodes=1)
    cfg["model"].update(width=32, layers=1, heads=4, ff_dim=64, time_dim=16)
    cfg["train"].update(
        steps=4,
        batch_size=4,
        warmup_steps=1,
        val_every=2,
        val_batches=1,
        log_every=1,
        precision="fp32",
    )
    n, length = 5, 40
    with h5py.File(cfg["dataset"], "w") as f:
        f["ep_len"] = np.full(n, length)
        f["ep_offset"] = np.arange(n) * length
        f.create_dataset(
            "pixels",
            (n * length, 224, 224, 3),
            dtype="uint8",
            chunks=(1, 224, 224, 3),
            compression="gzip",
        )
        f["action"] = np.tile(np.arange(n * length, dtype=np.float32)[:, None], (1, 5)) / 1000
        for key, width in (
            ("qpos", 15),
            ("qvel", 14),
            ("privileged_block_0_pos", 3),
            ("privileged_block_0_quat", 4),
        ):
            f[key] = np.zeros((n * length, width))
    m = make_split(cfg)
    with h5py.File(cfg["cache"], "w") as f:
        meta = {
            "manifest_id": m["manifest_id"],
            "source": m["source"],
            "action_stats": m["action_stats"],
            "encoder": {"test_fixture": True},
        }
        f.attrs["metadata"] = json.dumps(meta)
        stats = {"mean": [0] * 192, "scale": [1] * 192}
        f.attrs["latent_stats"] = json.dumps(stats)
        f.attrs["cache_id"] = digest(meta)
        f.attrs["complete"] = True
        for split, eps in m["episodes"].items():
            for ep in eps:
                g = f.create_group(f"{split}/{ep}")
                z = np.tile(np.arange(length, dtype=np.float32)[:, None], (1, 192))
                g["z"] = z
                g["action"] = np.tile(np.arange(length, dtype=np.float32)[:, None] / 100, (1, 5))
                g.attrs["complete"] = True
    return cfg
