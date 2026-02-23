#!/usr/bin/env python3
"""Entry point for the Weight Manager process."""

import argparse
import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.weight_manager import WeightManager


def main():
    parser = argparse.ArgumentParser(description="CUDA IPC Weight Manager")
    parser.add_argument("--model", default="mlp", choices=["mlp", "resnet18"])
    parser.add_argument("--device", default=None, help="CUDA device (default: from config)")
    parser.add_argument("--endpoint", default=None, help="ZMQ endpoint (default: from config)")
    parser.add_argument("--tp-rank", type=int, default=0, help="TP rank (default: 0)")
    parser.add_argument("--tp-world-size", type=int, default=1, help="TP world size (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible model init")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [WM] %(levelname)s %(message)s",
    )

    kwargs = {"model_name": args.model}
    if args.device:
        kwargs["device"] = args.device
    if args.endpoint:
        kwargs["endpoint"] = args.endpoint
    kwargs["tp_rank"] = args.tp_rank
    kwargs["tp_world_size"] = args.tp_world_size

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    wm = WeightManager(**kwargs)
    wm.load_model()
    wm.serve()


if __name__ == "__main__":
    main()
