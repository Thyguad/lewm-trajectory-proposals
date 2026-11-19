"""All planners return raw, bounded environment commands. No evaluator truth inputs."""

from __future__ import annotations

import time
import numpy as np
import torch

from .common import derived_seed, synchronize


def executable(normalized, stats):
    a = np.asarray(normalized).reshape(-1, 5, 5, 5)
    raw = a * np.asarray(stats["scale"]) + np.asarray(stats["mean"])
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite proposed actions")
    return np.clip(raw, -1, 1).astype(np.float32)


class Planner:
    def __init__(self, method, frozen, settings, seed, generator=None, checkpoint=None):
        self.method, self.frozen, self.settings, self.seed = method, frozen, settings, seed
        self.generator, self.checkpoint = generator, checkpoint
        self.calls = 0
        if method not in {"cem", "flow_direct", "flow_ranked"}:
            raise ValueError(method)
        if method != "cem" and (generator is None or checkpoint is None):
            raise ValueError("A generator checkpoint is required")

    def plan(self, current, goal):
        seed = derived_seed(self.seed, self.calls)
        self.calls += 1
        rng = np.random.default_rng(seed)
        f, s = self.frozen, self.settings
        before = (f.predictor_transitions, f.predictor_calls)
        synchronize(f.device)
        began = time.perf_counter()
        diagnostics = {}
        if self.method == "cem":
            mean, std = np.zeros((5, 5, 5)), np.ones((5, 5, 5))
            for _ in range(s["cem_iterations"]):
                normalized = rng.normal(mean, std, (s["cem_samples"], 5, 5, 5)).astype(np.float32)
                normalized[0] = mean
                actions = executable(normalized, f.stats)
                future = f.rollout(current, actions)
                costs = ((future[:, -1] - goal) ** 2).sum(-1)
                elite = normalized[np.argsort(costs)[: s["cem_elites"]]]
                mean, std = elite.mean(0), np.maximum(elite.std(0), s["cem_min_std"])
            # Preserve the current research CEM's final-mean execution convention.
            chosen = executable(mean[None], f.stats)[0]
            diagnostics["candidates_scored"] = s["cem_samples"] * s["cem_iterations"]
        else:
            stats = self.checkpoint["latent_stats"]
            zm, zs = np.asarray(stats["mean"]), np.asarray(stats["scale"])
            c = torch.as_tensor((current - zm) / zs, dtype=torch.float32, device=f.device)
            g = torch.as_tensor((goal - zm) / zs, dtype=torch.float32, device=f.device)
            count = 1 if self.method == "flow_direct" else s["candidates"]
            a, z = self.generator.sample(c, g, count=count, steps=s["flow_steps"], seed=seed)
            actions = executable(a.cpu().numpy(), f.stats)
            generated = z.cpu().numpy() * zs + zm
            if self.method == "flow_ranked":
                simulated = f.rollout(current, actions)
                costs = ((simulated[:, -1] - goal) ** 2).sum(-1)
                selected = int(np.argmin(costs))
                # Diagnostic only: never include generated-state agreement in selection.
                diagnostics.update(
                    selected_cost=float(costs[selected]),
                    selected_candidate=selected,
                    generated_predictor_mse=float(
                        ((generated[selected] - simulated[selected]) ** 2).mean()
                    ),
                    candidates_scored=count,
                )
            else:
                selected = 0
                diagnostics["candidates_scored"] = 0
            chosen = actions[selected]
            diagnostics.update(generated_candidates=count, flow_forward_calls=s["flow_steps"])
        synchronize(f.device)
        diagnostics.update(
            planning_seconds=time.perf_counter() - began,
            predictor_transitions=f.predictor_transitions - before[0],
            predictor_calls=f.predictor_calls - before[1],
        )
        return chosen.reshape(25, 5), diagnostics
