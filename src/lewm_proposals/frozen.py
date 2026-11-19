"""Checkpoint-compatible frozen LeWM with explicit preprocessing and action units."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn

from .common import file_hash, read_json
from .vendor.lewm import LeWM
from .vendor.module import Predictor, Embedder, MLP


def build_lewm(spec):
    from transformers import ViTConfig, ViTModel

    enc = spec["encoder"]
    if enc["size"] != "tiny" or enc.get("pretrained", False):
        raise ValueError("This loader supports the official LeWM tiny encoder checkpoint")
    # Same constructor parameters as stable_pretraining.backbone.utils.vit_hf('tiny').
    vit = ViTModel(
        ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            patch_size=enc["patch_size"],
            image_size=enc["image_size"],
        ),
        add_pooling_layer=False,
        use_mask_token=enc.get("use_mask_token", False),
    )

    def args(key):
        return {k: v for k, v in spec[key].items() if not k.startswith("_")}

    heads = {}
    for name in ("projector", "pred_proj"):
        kw = args(name)
        norm = kw.pop("norm_fn", None)
        if norm is not None and norm.get("_target_") != "torch.nn.BatchNorm1d":
            raise ValueError("Unexpected projector normalization")
        heads[name] = MLP(**kw, norm_fn=nn.BatchNorm1d if norm else nn.LayerNorm)
    return LeWM(vit, Predictor(**args("predictor")), Embedder(**args("action_encoder")), **heads)


def preprocess(frames, device):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (224, 224, 3):
        raise ValueError("Expected uint8 RGB frames [B,224,224,3]; no silent resizing")
    x = torch.as_tensor(frames.copy(), device=device).permute(0, 3, 1, 2).float() / 255.0
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((x - mean) / std)[:, None]


class FrozenLeWM:
    def __init__(self, weights, device, stats=None, score_batch_size=64):
        path = Path(weights).expanduser().resolve()
        if not path.is_file() or not (path.parent / "config.json").is_file():
            raise FileNotFoundError("Supply existing weights.pt and adjacent config.json")
        self.spec = read_json(path.parent / "config.json")
        if self.spec["action_encoder"]["input_dim"] != 25:
            raise ValueError("Checkpoint must have Cube 5x5 action blocks")
        self.model = build_lewm(self.spec)
        self.model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        self.model.eval().requires_grad_(False).to(device)
        self.device = torch.device(device)
        self.stats = stats
        self.score_batch_size = score_batch_size
        self.identity = {
            "weights_sha256": file_hash(path),
            "config_sha256": file_hash(path.parent / "config.json"),
            "preprocessing": "uint8 RGB224 / 255, ImageNet normalize; FP32",
        }
        self.encoder_frames = 0
        self.predictor_transitions = 0
        self.predictor_calls = 0

    @torch.inference_mode()
    def encode(self, frames, batch_size=64):
        out = []
        for i in range(0, len(frames), batch_size):
            part = frames[i : i + batch_size]
            z = self.model.encode({"pixels": preprocess(part, self.device)})["emb"][:, -1]
            out.append(z.float().cpu().numpy())
            self.encoder_frames += len(part)
        if not out:
            return np.empty((0, 192), dtype=np.float32)
        result = np.concatenate(out)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite encoder features")
        return result

    @torch.inference_mode()
    def rollout(self, current, actions):
        """Raw bounded commands [K,H,5,5] -> raw LeWM latents [K,H,192]."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 4 or actions.shape[2:] != (5, 5):
            raise ValueError("Expected actions [K,H,5,5]")
        if not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
            raise ValueError(
                "Predictor must score the exact finite, clipped commands that will execute"
            )
        if self.stats is None:
            raise ValueError("Action statistics are required for rollout")
        mean, scale = np.asarray(self.stats["mean"]), np.asarray(self.stats["scale"])
        normalized = (
            ((actions - mean) / scale)
            .astype(np.float32)
            .reshape(len(actions), actions.shape[1], 25)
        )
        outputs = []
        max_context = self.spec["predictor"]["num_frames"]
        for start in range(0, len(actions), self.score_batch_size):
            a = torch.as_tensor(
                normalized[start : start + self.score_batch_size], device=self.device
            )
            z = (
                torch.as_tensor(current, dtype=torch.float32, device=self.device)
                .reshape(1, 1, -1)
                .expand(len(a), 1, -1)
            )
            future = []
            for h in range(a.shape[1]):
                ctx_a = a[:, max(0, h - max_context + 1) : h + 1]
                nxt = self.model.predict(z[:, -max_context:], self.model.action_encoder(ctx_a))[
                    :, -1:
                ]
                future.append(nxt)
                z = torch.cat((z, nxt), dim=1)
                self.predictor_transitions += len(a)
                self.predictor_calls += 1
            outputs.append(torch.cat(future, dim=1).cpu().numpy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite predictor rollout")
        return result
