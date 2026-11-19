import numpy as np
import torch

from lewm_proposals.flow import JointFlow
from lewm_proposals.train import train


def test_both_output_heads_learn_and_sampling_is_repeatable(small_config):
    from lewm_proposals.data import WindowStore

    model = JointFlow(small_config["model"])
    batch = WindowStore(small_config["cache"], "train").sample(4, np.random.default_rng(5), "cpu")
    loss, _ = model.loss(batch, torch.Generator().manual_seed(3))
    loss.backward()
    assert model.a_out.weight.grad.norm() > 0
    assert model.z_out.weight.grad.norm() > 0
    model.eval()
    a, z = model.sample(batch["current"][0], batch["goal"][0], count=1, steps=3, seed=42)
    aa, zz = model.sample(batch["current"][0], batch["goal"][0], count=4, steps=3, seed=42)
    torch.testing.assert_close(a[0], aa[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z[0], zz[0], atol=1e-5, rtol=1e-5)
    assert not torch.equal(aa[0], aa[1])


def test_checkpoint_resume_matches_uninterrupted_updates(small_config, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(small_config, full)
    train(small_config, resumed, stop_after=2)
    train(small_config, resumed, resume=True)
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
