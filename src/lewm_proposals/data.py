"""Whole-episode splits, read-only source HDF5, resumable latent cache and windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  register source compression filters
import numpy as np
import torch

from .common import digest, read_json, write_json


class CubeData:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.f = h5py.File(self.path, "r")
        required = {
            "pixels",
            "action",
            "ep_len",
            "ep_offset",
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        }
        if missing := required - set(self.f):
            self.f.close()
            raise ValueError(f"Missing dataset keys: {sorted(missing)}")
        self.lengths = self.f["ep_len"][:].astype(np.int64)
        self.offsets = self.f["ep_offset"][:].astype(np.int64)
        if self.f["pixels"].shape[1:] != (224, 224, 3) or self.f["action"].shape[1:] != (5,):
            raise ValueError("Expected Cube RGB224 and five-dimensional actions")
        if self.f["pixels"].dtype != np.uint8:
            raise ValueError("Pixel dtype must be uint8")
        n = len(self.f["action"])
        if len(self.offsets) != len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("Invalid episode metadata")
        if np.any(self.offsets < 0) or np.any(self.offsets + self.lengths > n):
            raise ValueError("Episode offsets out of bounds")
        if np.any(self.offsets[1:] < self.offsets[:-1] + self.lengths[:-1]):
            raise ValueError("Episode intervals overlap or are unordered")
        if any(len(self.f[k]) != n for k in required - {"ep_len", "ep_offset"}):
            raise ValueError("Dataset columns have unequal row counts")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.f.close()

    def identity(self):
        # Explicitly a sampled fingerprint, not a claim of hashing the ~100GB file.
        h = hashlib.sha256(self.lengths.tobytes() + self.offsets.tobytes())
        for row in sorted({0, len(self.f["action"]) // 2, len(self.f["action"]) - 1}):
            for key in ("pixels", "action", "qpos", "qvel"):
                h.update(np.asarray(self.f[key][row]).tobytes())
        return {
            "kind": "sampled-content-v1",
            "file_bytes": self.path.stat().st_size,
            "rows": len(self.f["action"]),
            "episodes": len(self.lengths),
            "fingerprint": h.hexdigest(),
        }

    def action_stats(self):
        # Preserve the frozen checkpoint's original all-source StandardScaler convention.
        n, total, square = 0, np.zeros(5), np.zeros(5)
        for start in range(0, len(self.f["action"]), 100000):
            a = self.f["action"][start : start + 100000].astype(np.float64)
            a = a[np.isfinite(a).all(axis=1)]
            n += len(a)
            total += a.sum(0)
            square += (a * a).sum(0)
        if n == 0:
            raise ValueError("No finite actions")
        mean = total / n
        scale = np.sqrt(np.maximum(square / n - mean**2, 0))
        scale[scale < 1e-12] = 1.0
        return {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "rows": n,
            "source": "all original pretraining-data actions; fixed for all methods",
        }

    def frames(self, ep):
        off, length = int(self.offsets[ep]), int(self.lengths[ep])
        return off, length

    def initial_and_goal(self, case):
        row = int(self.offsets[case["episode"]]) + case["start"]
        end = row + case["goal_offset"]
        return (
            {"qpos": self.f["qpos"][row], "qvel": self.f["qvel"][row]},
            {
                "pos": self.f["privileged_block_0_pos"][end],
                "quat": self.f["privileged_block_0_quat"][end],
            },
            self.f["pixels"][end],
            self.f["pixels"][row],
        )


def validate_manifest(m):
    expected = m.get("manifest_id")
    if expected != digest({k: v for k, v in m.items() if k != "manifest_id"}):
        raise ValueError("Manifest integrity mismatch")
    sets = [set(m["episodes"][s]) for s in ("train", "val", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Episode leakage between splits")
    if any(len(m["episodes"][s]) != len(set(m["episodes"][s])) for s in m["episodes"]):
        raise ValueError("Duplicate episode IDs")
    if any(ep < 0 or ep >= m["source"]["episodes"] for group in sets for ep in group):
        raise ValueError("Invalid episode ID")


def make_split(cfg):
    if Path(cfg["manifest"]).exists():
        raise FileExistsError("Manifest exists; reuse it or choose a new output path")
    with CubeData(cfg["dataset"]) as data:
        split = cfg["split"]
        excluded = set(split.get("exclude_episodes", []))
        eligible = np.array([i for i, n in enumerate(data.lengths) if n > 25 and i not in excluded])
        rng = np.random.default_rng(split["seed"])
        rng.shuffle(eligible)
        counts = [int(split[f"{s}_episodes"]) for s in ("train", "val", "test")]
        if min(counts) < 1 or sum(counts) > len(eligible):
            raise ValueError(f"Requested {counts}; only {len(eligible)} eligible episodes")
        groups, start = {}, 0
        for s, n in zip(("train", "val", "test"), counts):
            groups[s] = sorted(eligible[start : start + n].tolist())
            start += n
        m = {
            "schema": "lewm-proposals-split-v1",
            "source": data.identity(),
            "split_config": split,
            "episodes": groups,
            "action_stats": data.action_stats(),
            "horizon_steps": 25,
            "scope": "Held out from new generator training; pretrained LeWM may have seen all episodes",
        }
        m["manifest_id"] = digest(m)
        validate_manifest(m)
        write_json(cfg["manifest"], m)
    return m


def encode_cache(cfg, frozen, *, resume=False):
    manifest = read_json(cfg["manifest"])
    validate_manifest(manifest)
    path = Path(cfg["cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "lewm-latents-v1",
        "manifest_id": manifest["manifest_id"],
        "source": manifest["source"],
        "encoder": frozen.identity,
        "dtype": "float16",
        "action_stats": manifest["action_stats"],
    }
    mode = "a" if resume else "x"
    with CubeData(cfg["dataset"]) as source, h5py.File(path, mode) as out:
        if source.identity() != manifest["source"]:
            raise ValueError("Source dataset differs from the split manifest")
        if "metadata" in out.attrs:
            if json.loads(out.attrs["metadata"]) != meta:
                raise ValueError("Cannot resume a cache made with different data, split or weights")
        else:
            out.attrs["metadata"] = json.dumps(meta)
            out.attrs["complete"] = False
        for split, episodes in manifest["episodes"].items():
            parent = out.require_group(split)
            for i, ep in enumerate(episodes):
                name = str(ep)
                if name in parent and bool(parent[name].attrs.get("complete", False)):
                    continue
                if name in parent:
                    del parent[name]  # only the incomplete, task-created cache group
                group = parent.create_group(name)
                off, length = source.frames(ep)
                z = group.create_dataset("z", (length, 192), dtype="float16")
                bs = cfg["encode_batch_size"]
                for j in range(0, length, bs):
                    z[j : j + bs] = frozen.encode(
                        source.f["pixels"][off + j : off + min(j + bs, length)], bs
                    )
                group.create_dataset(
                    "action", data=source.f["action"][off : off + length].astype(np.float32)
                )
                group.attrs["complete"] = True
                out.flush()
                if (i + 1) % 50 == 0 or i + 1 == len(episodes):
                    print(f"cache {split}: {i + 1}/{len(episodes)} episodes", flush=True)
        # Feature scaling belongs to G only and is fitted on TRAIN episodes only.
        count, sums, squares = 0, np.zeros(192), np.zeros(192)
        for group in out["train"].values():
            z = group["z"][:].astype(np.float64)
            if not np.isfinite(z).all():
                raise FloatingPointError("Nonfinite cached features")
            count += len(z)
            sums += z.sum(0)
            squares += (z * z).sum(0)
        mean = sums / count
        scale = np.sqrt(np.maximum(squares / count - mean * mean, 0)).clip(1e-6)
        out.attrs["latent_stats"] = json.dumps({"mean": mean.tolist(), "scale": scale.tolist()})
        out.attrs["cache_id"] = digest(
            {**meta, "latent_stats": json.loads(out.attrs["latent_stats"])}
        )
        out.attrs["complete"] = True
        out.flush()


class WindowStore:
    """Load compact latent/action arrays only; never keep RGB frames in training memory."""

    def __init__(self, path, split, horizon=5, block=5):
        self.horizon, self.block = horizon, block
        self.items, self.starts, self.cumulative = [], [], []
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete", False):
                raise ValueError("Incomplete cache; finish encode --resume first")
            self.meta = json.loads(f.attrs["metadata"])
            self.cache_id = str(f.attrs["cache_id"])
            self.latent_stats = json.loads(f.attrs["latent_stats"])
            self.action_stats = self.meta["action_stats"]
            zm, zs = np.array(self.latent_stats["mean"]), np.array(self.latent_stats["scale"])
            am, ast = np.array(self.action_stats["mean"]), np.array(self.action_stats["scale"])
            for ep in sorted(f[split], key=int):
                group = f[split][ep]
                if not group.attrs.get("complete", False):
                    raise ValueError("Incomplete episode cache")
                z = ((group["z"][:].astype(np.float32) - zm) / zs).astype(np.float32)
                a = ((group["action"][:] - am) / ast).astype(np.float32)
                if not np.isfinite(z).all():
                    raise FloatingPointError("Invalid cached states")
                span = horizon * block
                bad = ~np.isfinite(a).all(axis=1)
                prefix = np.r_[0, np.cumsum(bad)]
                starts = np.arange(max(0, len(z) - span))
                starts = starts[(prefix[starts + span] - prefix[starts]) == 0]
                if len(starts):
                    self.items.append((z, a))
                    self.starts.append(starts)
                    self.cumulative.append(
                        len(starts) + (self.cumulative[-1] if self.cumulative else 0)
                    )
        if not self.items:
            raise ValueError(f"No valid {split} windows")
        self.cumulative = np.array(self.cumulative)

    def sample(self, batch_size, rng, device):
        indices = rng.integers(0, self.cumulative[-1], size=batch_size)
        current, goal, future, actions = [], [], [], []
        for idx in indices:
            ep = int(np.searchsorted(self.cumulative, idx, side="right"))
            start = self.starts[ep][idx - (self.cumulative[ep - 1] if ep else 0)]
            z, a = self.items[ep]
            end = start + self.horizon * self.block
            current.append(z[start])
            goal.append(z[end])
            future.append(z[start + self.block : end + 1 : self.block])
            actions.append(a[start:end].reshape(self.horizon, -1))
        return {
            k: torch.as_tensor(np.stack(v), device=device)
            for k, v in {
                "current": current,
                "goal": goal,
                "future": future,
                "actions": actions,
            }.items()
        }


def evaluation_cases(source, manifest, split, count, seed):
    episodes = np.array(manifest["episodes"][split])
    if count > len(episodes):
        raise ValueError("Requested more evaluation cases than held-out episodes")
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(episodes)[:count]
    cases = []
    for ep in chosen:
        start = int(rng.integers(0, int(source.lengths[ep]) - 25))
        cases.append(
            {
                "case_id": f"ep{ep}-s{start}-g25",
                "episode": int(ep),
                "start": start,
                "goal_offset": 25,
            }
        )
    return cases
