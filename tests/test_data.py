import h5py
import numpy as np
import pytest

from lewm_proposals.common import digest, read_json
from lewm_proposals.data import WindowStore, validate_manifest, encode_cache


def test_whole_episode_split_and_window_alignment(small_config):
    cfg = small_config
    m = read_json(cfg["manifest"])
    validate_manifest(m)
    train, val, test = (set(m["episodes"][s]) for s in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    data = WindowStore(cfg["cache"], "train")
    batch = data.sample(32, np.random.default_rng(9), "cpu")
    start = batch["current"][:, 0].numpy()
    np.testing.assert_array_equal(batch["goal"][:, 0], start + 25)
    np.testing.assert_array_equal(batch["future"][:, :, 0], start[:, None] + np.arange(5, 26, 5))
    mean, scale = np.array(data.action_stats["mean"]), np.array(data.action_stats["scale"])
    raw = batch["actions"].numpy().reshape(32, 25, 5) * scale + mean
    np.testing.assert_allclose(raw[:, :, 0], (start[:, None] + np.arange(25)) / 100, atol=1e-7)


def test_leakage_and_incomplete_cache_rejected(small_config):
    m = read_json(small_config["manifest"])
    m["episodes"]["test"] = [m["episodes"]["train"][0]]
    m["manifest_id"] = digest({k: v for k, v in m.items() if k != "manifest_id"})
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(m)
    with h5py.File(small_config["cache"], "a") as f:
        f.attrs["complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        WindowStore(small_config["cache"], "train")


def test_cache_resume_identity_and_train_only_normalization(small_config, tmp_path):
    class Encoder:
        identity = {"fixture": "deterministic-rgb"}

        def encode(self, x, batch_size):
            return np.repeat(x[:, 0, 0, 0, None].astype(np.float32), 192, axis=1)

    cfg = small_config
    cfg["cache"] = str(tmp_path / "new-cache.h5")
    m = read_json(cfg["manifest"])
    # The source fingerprint samples first/middle/last; modify before regenerating identity.
    from lewm_proposals.data import CubeData
    from lewm_proposals.common import write_json

    with h5py.File(cfg["dataset"], "a") as f:
        for split, eps in m["episodes"].items():
            for ep in eps:
                off, length = int(f["ep_offset"][ep]), int(f["ep_len"][ep])
                f["pixels"][off : off + length] = 10 if split == "train" else 100
    with CubeData(cfg["dataset"]) as source:
        m["source"] = source.identity()
    m["manifest_id"] = digest({k: v for k, v in m.items() if k != "manifest_id"})
    write_json(cfg["manifest"], m, replace=True)
    encode_cache(cfg, Encoder())
    encode_cache(cfg, Encoder(), resume=True)
    data = WindowStore(cfg["cache"], "train")
    assert data.latent_stats["mean"] == [10.0] * 192
    encoder = Encoder()
    encoder.identity = {"fixture": "changed"}
    with pytest.raises(ValueError, match="different"):
        encode_cache(cfg, encoder, resume=True)
