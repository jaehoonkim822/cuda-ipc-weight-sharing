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
    parser.add_argument("--socket", default=None, help="UDS path (default: from config)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [Worker] %(levelname)s %(message)s",
    )

    kwargs = {"model_name": args.model}
    if args.device:
        kwargs["device"] = args.device
    if args.socket:
        kwargs["socket_path"] = args.socket

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
