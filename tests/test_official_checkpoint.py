"""Optional real-checkpoint parity check; no mock weights count as passing this test."""

import os
import numpy as np
import pytest
import torch

from lewm_proposals.frozen import FrozenLeWM, preprocess


@pytest.mark.skipif(
    not os.environ.get("LEWM_TEST_WEIGHTS"), reason="Set LEWM_TEST_WEIGHTS to official weights.pt"
)
def test_rollout_matches_upstream_and_parameters_remain_frozen():
    torch.set_num_threads(2)
    stats = {"mean": [0.02] * 5, "scale": [0.2] * 5}
    frozen = FrozenLeWM(os.environ["LEWM_TEST_WEIGHTS"], "cpu", stats)
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    actions = rng.uniform(-0.3, 0.3, (2, 5, 5, 5)).astype(np.float32)
    z = frozen.encode(image)[0]
    ours = frozen.rollout(z, actions)
    scaled = ((actions - np.array(stats["mean"])) / np.array(stats["scale"])).astype(np.float32)
    pixels = preprocess(image, "cpu")[:, None].expand(1, 2, 1, 3, 224, 224)
    with torch.inference_mode():
        reference = frozen.model.rollout(
            {"pixels": pixels}, torch.from_numpy(scaled.reshape(1, 2, 5, 25))
        )["predicted_emb"][0, :, 1:]
    np.testing.assert_allclose(ours, reference.numpy(), atol=2e-5, rtol=2e-5)
    assert not frozen.model.training
    assert all(not p.requires_grad and p.grad is None for p in frozen.model.parameters())
