"""Goal-conditioned joint state/action conditional flow matching.

Only this module is trained. The frozen LeWM predictor is not in the training graph.
States and actions have separate projections and dimension-normalized losses.
"""

from __future__ import annotations

import math
import torch
from torch import nn


class JointFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.horizon = cfg["horizon"]
        self.z_dim = cfg["latent_dim"]
        self.a_dim = cfg["action_dim"] * cfg["action_block"]
        w = cfg["width"]
        self.z_in, self.a_in = nn.Linear(self.z_dim, w), nn.Linear(self.a_dim, w)
        self.z_out, self.a_out = nn.Linear(w, self.z_dim), nn.Linear(w, self.a_dim)
        self.types = nn.Embedding(4, w)  # current, goal, action, generated state
        self.time = nn.Sequential(nn.Linear(cfg["time_dim"], w), nn.SiLU(), nn.Linear(w, w))
        # Independent layer initialization: do not leave cloned layers identical.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    w,
                    cfg["heads"],
                    cfg["ff_dim"],
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg["layers"])
            ]
        )
        self.norm = nn.LayerNorm(w)
        pos = torch.arange(2 + 2 * self.horizon, dtype=torch.float32)[:, None]
        freq = torch.exp(-math.log(10000) * torch.arange(0, w, 2) / w)
        pe = torch.zeros(2 + 2 * self.horizon, w)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        self.register_buffer("positions", pe[None])
        self.register_buffer(
            "time_freq",
            torch.exp(
                -math.log(10000) * torch.arange(cfg["time_dim"] // 2) / (cfg["time_dim"] // 2)
            ),
        )

    def forward(self, noisy_a, noisy_z, tau, current, goal):
        b, h, _ = noisy_a.shape
        if h != self.horizon or noisy_z.shape != (b, h, self.z_dim):
            raise ValueError("Unexpected trajectory shape")
        pair = torch.stack(
            (self.a_in(noisy_a) + self.types.weight[2], self.z_in(noisy_z) + self.types.weight[3]),
            dim=2,
        )
        context = torch.stack(
            (self.z_in(current) + self.types.weight[0], self.z_in(goal) + self.types.weight[1]),
            dim=1,
        )
        x = torch.cat((context, pair.flatten(1, 2)), dim=1)
        angles = tau[:, None] * self.time_freq[None] * 1000
        t = self.time(torch.cat((angles.cos(), angles.sin()), dim=-1))
        x = x + self.positions + t[:, None]
        for layer in self.layers:
            x = layer(x)  # Bidirectional attention; condition tokens contain no unknown future.
        x = self.norm(x[:, 2:])
        return self.a_out(x[:, 0::2]), self.z_out(x[:, 1::2])

    def loss(self, batch, generator, state_weight=1.0, action_weight=1.0):
        # CPU RNG makes the stochastic data stream portable across CPU/CUDA/MPS.
        a, z = batch["actions"], batch["future"]
        noise_a = torch.randn(a.shape, generator=generator).to(a.device)
        noise_z = torch.randn(z.shape, generator=generator).to(z.device)
        tau = torch.rand(len(a), generator=generator).to(a.device)
        s = tau[:, None, None]
        va, vz = self(
            (1 - s) * noise_a + s * a,
            (1 - s) * noise_z + s * z,
            tau,
            batch["current"],
            batch["goal"],
        )
        la = (va.float() - (a - noise_a)).square().mean()
        lz = (vz.float() - (z - noise_z)).square().mean()
        return action_weight * la + state_weight * lz, {"action_loss": la, "state_loss": lz}

    @torch.inference_mode()
    def sample(self, current, goal, *, count, steps, seed):
        if count <= 0 or steps <= 0:
            raise ValueError("count and steps must be positive")
        # Each candidate owns a substream: candidate 0 is identical for K=1 and K=16.
        from .common import derived_seed

        all_a, all_z = [], []
        for k in range(count):
            g = torch.Generator().manual_seed(derived_seed(seed, k))
            all_a.append(torch.randn(self.horizon, self.a_dim, generator=g))
            all_z.append(torch.randn(self.horizon, self.z_dim, generator=g))
        a, z = torch.stack(all_a).to(current.device), torch.stack(all_z).to(current.device)
        c, goal = current.reshape(1, -1).expand(count, -1), goal.reshape(1, -1).expand(count, -1)
        for i in range(steps):
            tau = torch.full((count,), i / steps, device=current.device)
            va, vz = self(a, z, tau, c, goal)
            a, z = a + va / steps, z + vz / steps
        if not torch.isfinite(a).all() or not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite generated trajectory")
        return a, z
