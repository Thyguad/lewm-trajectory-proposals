from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Stage-one frozen-LeWM trajectory proposal experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "split", "encode", "train", "evaluate", "doctor"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name in ("train", "evaluate"):
            p.add_argument("--output", required=True)
        if name in ("encode", "train", "evaluate"):
            p.add_argument("--resume", action="store_true")
        if name == "train":
            p.add_argument(
                "--stop-after",
                type=int,
                help="Pause after N additional steps without changing the schedule",
            )
        if name == "evaluate":
            p.add_argument("--checkpoint")
        if name == "doctor":
            p.add_argument("--load-model", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        from .evaluate import summarize

        print(json.dumps(summarize(args.run), ensure_ascii=False, indent=2))
        return
    from .common import load_config, resolve_device, runtime_metadata

    cfg = load_config(args.config)
    if args.command == "doctor":
        result = {
            "runtime": runtime_metadata(),
            "device": str(resolve_device(cfg["device"])),
            "inputs": {
                k: bool(cfg.get(k) and Path(cfg[k]).expanduser().is_file())
                for k in ("dataset", "weights")
            },
        }
        if args.load_model:
            from .frozen import FrozenLeWM

            model = FrozenLeWM(cfg["weights"], resolve_device(cfg["device"]))
            result["frozen"] = model.identity
            result["frozen_parameters"] = sum(p.numel() for p in model.model.parameters())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not all(result["inputs"].values()):
            raise SystemExit("Missing local dataset/weights; set paths before running experiments")
    elif args.command == "inspect":
        from .data import CubeData

        with CubeData(cfg["dataset"]) as source:
            print(
                json.dumps(
                    {
                        "identity": source.identity(),
                        "min_episode_steps": int(source.lengths.min()),
                        "max_episode_steps": int(source.lengths.max()),
                        "columns": list(source.f),
                    },
                    indent=2,
                )
            )
    elif args.command == "split":
        from .data import make_split

        m = make_split(cfg)
        print(
            json.dumps(
                {
                    "manifest_id": m["manifest_id"],
                    "counts": {k: len(v) for k, v in m["episodes"].items()},
                },
                indent=2,
            )
        )
    elif args.command == "encode":
        import torch
        from .data import encode_cache
        from .frozen import FrozenLeWM

        torch.set_num_threads(cfg.get("cpu_threads", 4))
        frozen = FrozenLeWM(cfg["weights"], resolve_device(cfg["device"]))
        encode_cache(cfg, frozen, resume=args.resume)
    elif args.command == "train":
        from .train import train

        train(cfg, args.output, resume=args.resume, stop_after=args.stop_after)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(cfg, args.output, args.checkpoint, resume=args.resume)


if __name__ == "__main__":
    main()
