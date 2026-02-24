#!/usr/bin/env python3
"""Entry point for an Inference Worker process."""

import argparse
import logging
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.inference_worker import InferenceWorker


def main():
    parser = argparse.ArgumentParser(description="CUDA IPC Inference Worker")
    parser.add_argument("--model", default="mlp", choices=["mlp", "resnet18"])
    parser.add_argument("--device", default=None, help="CUDA device (default: from config)")
    parser.add_argument("--endpoint", default=None, help="ZMQ endpoint for single WM (default: from config)")
    parser.add_argument("--endpoints", default=None, help="Comma-separated ZMQ endpoints for TP mode")
    parser.add_argument("--distributed-tp", action="store_true", help="Enable distributed TP mode (1:1 WM-Worker)")
    parser.add_argument("--tp-rank", type=int, default=0, help="This worker's TP rank")
    parser.add_argument("--tp-world-size", type=int, default=1, help="Total number of TP ranks")
    parser.add_argument("--master-addr", default="127.0.0.1", help="NCCL master address")
    parser.add_argument("--master-port", default="29500", help="NCCL master port")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [Worker] %(levelname)s %(message)s",
    )

    kwargs = {"model_name": args.model}
    if args.device:
        kwargs["device"] = args.device
    if args.endpoints:
        kwargs["endpoint"] = [ep.strip() for ep in args.endpoints.split(",")]
    elif args.endpoint:
        kwargs["endpoint"] = args.endpoint

    if args.distributed_tp:
        import os
        os.environ.setdefault("MASTER_ADDR", args.master_addr)
        os.environ.setdefault("MASTER_PORT", args.master_port)
        os.environ["RANK"] = str(args.tp_rank)
        os.environ["WORLD_SIZE"] = str(args.tp_world_size)
        kwargs["distributed_tp"] = True

    worker = InferenceWorker(**kwargs)
    worker.connect_and_load()

    sample_input = worker._sample_input_fn(worker._device)
    output = worker.infer(sample_input)
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:       {output}")

    # Show data pointers for verification
    ptrs = worker.get_data_ptrs()
    print(f"\nData pointers ({len(ptrs)} tensors):")
    for name, ptr in ptrs.items():
        print(f"  {name}: 0x{ptr:x}")

    worker.cleanup()


if __name__ == "__main__":
    main()
