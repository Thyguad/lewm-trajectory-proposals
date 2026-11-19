import numpy as np
import torch

from lewm_proposals.planning import Planner, executable


class World:
    device = torch.device("cpu")
    stats = {"mean": [0] * 5, "scale": [1] * 5}
    predictor_transitions = 0
    predictor_calls = 0

    def rollout(self, current, actions):
        assert np.abs(actions).max() <= 1
        self.predictor_transitions += len(actions) * 5
        self.predictor_calls += 5
        self.last_actions = actions.copy()
        out = np.zeros((len(actions), 5, 192), dtype=np.float32)
        out[:, :, 0] = actions[:, 0, 0, 0, None]
        return out


class Generator:
    def sample(self, current, goal, *, count, steps, seed):
        a = torch.full((count, 5, 25), 0.9)
        if count > 1:
            a[1] = 0.1
        z = torch.zeros(count, 5, 192)
        z[0, :, 0] = 0.1  # Lie: candidate 0 claims to reach the goal.
        if count > 1:
            z[1, :, 0] = 100
        return a, z


def test_ranking_uses_frozen_rollout_not_generated_states(small_config):
    s = small_config["eval"]
    s["candidates"] = 2
    world = World()
    ckpt = {"latent_stats": {"mean": [0] * 192, "scale": [1] * 192}}
    current, goal = np.zeros(192), np.zeros(192)
    goal[0] = 0.1
    ranked = Planner("flow_ranked", world, s, 42, Generator(), ckpt)
    actions, info = ranked.plan(current, goal)
    np.testing.assert_allclose(actions, 0.1)
    assert info["selected_candidate"] == 1
    direct = Planner("flow_direct", world, s, 42, Generator(), ckpt)
    actions, info = direct.plan(current, goal)
    np.testing.assert_allclose(actions, 0.9)
    assert info["predictor_transitions"] == 0


def test_action_bounds_and_cem_budget(small_config):
    stats = {"mean": [0.2] * 5, "scale": [0.1] * 5}
    x = np.zeros((1, 5, 25))
    np.testing.assert_allclose(executable(x, stats), 0.2)
    assert executable(x + 100, stats).max() == 1
    s = small_config["eval"]
    s.update(cem_samples=12, cem_iterations=2, cem_elites=3)
    world = World()
    p = Planner("cem", world, s, 42)
    plan, info = p.plan(np.zeros(192), np.zeros(192))
    assert plan.shape == (25, 5)
    assert np.abs(plan).max() <= 1
    assert info["predictor_transitions"] == 12 * 2 * 5
