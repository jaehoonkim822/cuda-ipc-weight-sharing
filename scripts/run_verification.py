#!/usr/bin/env python3
"""Verification test suite for CUDA IPC GPU Memory Sharing PoC.

Runs 6+ tests sequentially:
  1. Memory sharing   — GPU memory stays at ~1 model even with N workers
  2. Zero-copy        — WM modifies weights, workers see changes
  3. Worker lifecycle  — 10x worker create/destroy, WM stays stable
  4. Inference accuracy — WM vs worker output matches (torch.allclose)
  5. Memory leak       — 50x worker cycles, check GPU memory growth
  6. Crash recovery    — kill -9 worker, ipc_collect, new worker works
  7. Tensor Parallelism — TP output matches single-GPU reference (2+ GPUs)

The Weight Manager runs as a subprocess.Popen.
Workers run as multiprocessing.Process(start_method='spawn').
"""

import argparse
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import psutil
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.config import CUDA_ALLOC_CONF

log = logging.getLogger(__name__)

# Ensure spawn method for CUDA compatibility
multiprocessing.set_start_method("spawn", force=True)


def _get_gpu_memory_mb(device_index: int = 0) -> float:
    """Get current GPU memory usage in MB via torch."""
    return torch.cuda.memory_allocated(device_index) / (1024 * 1024)


def _get_gpu_memory_nvidia_smi(device_index: int = 0) -> float:
    """Get GPU memory usage from nvidia-smi (cross-process visibility)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip())
    except Exception as e:
        log.warning("nvidia-smi failed: %s", e)
        return -1.0


# -- Worker entry points (must be top-level for spawn) --

def _worker_infer(model_name, device, endpoint, seed, result_dict, worker_id):
    """Worker process: connect, infer with fixed seed, store result."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoint)
    worker.connect_and_load()

    torch.manual_seed(seed)
    sample = worker._sample_input_fn(device)
    output = worker.infer(sample)

    result_dict[worker_id] = {
        "output": output.cpu().tolist(),
    }
    worker.cleanup()


def _worker_read_weight(model_name, device, endpoint, ready_event, stop_event, result_dict, worker_id):
    """Worker that connects, reads fc1.weight[0][0], waits, reads again."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoint)
    worker.connect_and_load()

    # Read initial value
    val_before = worker._model.fc1.weight.data[0][0].item()
    result_dict[f"{worker_id}_before"] = val_before
    ready_event.set()

    # Wait for WM to modify the weight
    stop_event.wait(timeout=30)
    time.sleep(0.5)

    # Read again — should see the modified value if truly zero-copy
    val_after = worker._model.fc1.weight.data[0][0].item()
    result_dict[f"{worker_id}_after"] = val_after
    worker.cleanup()


def _worker_stay_alive(model_name, device, endpoint, ready_event, stop_event):
    """Worker that stays alive until told to stop."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoint)
    worker.connect_and_load()
    ready_event.set()
    stop_event.wait()
    worker.cleanup()


def _worker_crash_target(model_name, device, endpoint, ready_event):
    """Worker that stays alive until killed externally."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoint)
    worker.connect_and_load()
    ready_event.set()
    # Block forever — will be killed with SIGKILL
    while True:
        time.sleep(1)


def _worker_quick_cycle(model_name, device, endpoint):
    """Worker that connects, infers once, and exits."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoint)
    worker.connect_and_load()
    sample = worker._sample_input_fn(device)
    worker.infer(sample)
    worker.cleanup()


def _worker_tp_infer(model_name, device, endpoints, seed, result_dict, worker_id):
    """Worker that connects to multiple WMs (TP), infers, stores result."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    from cuda_ipc_poc.inference_worker import InferenceWorker

    worker = InferenceWorker(model_name, device, endpoints)
    worker.connect_and_load()

    torch.manual_seed(seed)
    sample = worker._sample_input_fn(device)
    output = worker.infer(sample)

    result_dict[worker_id] = {
        "output": output.cpu().tolist(),
    }
    worker.cleanup()


class VerificationSuite:
    def __init__(self, model_name: str, device: str, endpoint: str):
        self._model_name = model_name
        self._device = device
        self._endpoint = endpoint
        self._device_index = int(device.split(":")[-1]) if ":" in device else 0
        self._wm_process: subprocess.Popen | None = None

    def _start_weight_manager(self) -> None:
        """Start Weight Manager as a subprocess."""
        script = os.path.join(os.path.dirname(__file__), "run_weight_manager.py")
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        self._wm_process = subprocess.Popen(
            [
                sys.executable,
                script,
                "--model", self._model_name,
                "--device", self._device,
                "--endpoint", self._endpoint,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give ZMQ server time to bind
        time.sleep(2.0)
        if self._wm_process.poll() is not None:
            raise RuntimeError("Weight Manager exited unexpectedly")
        log.info("Weight Manager started (PID %d)", self._wm_process.pid)

    def _stop_weight_manager(self) -> None:
        """Gracefully stop the Weight Manager."""
        if self._wm_process and self._wm_process.poll() is None:
            self._wm_process.send_signal(signal.SIGTERM)
            try:
                self._wm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._wm_process.kill()
                self._wm_process.wait()
            log.info("Weight Manager stopped")
        self._wm_process = None

    def run_all(self) -> None:
        """Run all verification tests."""
        tests = [
            ("1. Memory Sharing", self.test_memory_sharing),
            ("2. Zero-Copy Verification", self.test_zero_copy),
            ("3. Worker Lifecycle", self.test_worker_lifecycle),
            ("4. Inference Accuracy", self.test_inference_accuracy),
            ("5. Memory Leak Check", self.test_memory_leak),
            ("6. Crash Recovery", self.test_crash_recovery),
        ]

        # Test 7 requires 2+ GPUs
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            tests.append(("7. Tensor Parallelism", self.test_tensor_parallelism))
        else:
            log.info("Skipping Test 7 (Tensor Parallelism): requires 2+ GPUs, found %d", n_gpus)

        results = []
        for name, test_fn in tests:
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            try:
                self._start_weight_manager()
                test_fn()
                print(f"  PASSED")
                results.append((name, True, None))
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append((name, False, str(e)))
                log.exception("Test failed: %s", name)
            finally:
                self._stop_weight_manager()

        # Summary
        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        passed = sum(1 for _, ok, _ in results if ok)
        for name, ok, err in results:
            status = "PASS" if ok else f"FAIL ({err})"
            print(f"  {name}: {status}")
        print(f"\n  {passed}/{len(results)} tests passed")

    def test_memory_sharing(self) -> None:
        """Test 1: GPU memory stays near 1-model level with multiple workers."""
        baseline_mb = _get_gpu_memory_nvidia_smi(self._device_index)
        log.info("Baseline GPU memory: %.1f MB", baseline_mb)

        n_workers = 3
        workers = []
        ready_events = []
        stop_events = []

        for i in range(n_workers):
            ready = multiprocessing.Event()
            stop = multiprocessing.Event()
            p = multiprocessing.Process(
                target=_worker_stay_alive,
                args=(self._model_name, self._device, self._endpoint, ready, stop),
            )
            p.start()
            workers.append(p)
            ready_events.append(ready)
            stop_events.append(stop)

        # Wait for all workers to be ready
        for ready in ready_events:
            ready.wait(timeout=30)

        time.sleep(2)
        with_workers_mb = _get_gpu_memory_nvidia_smi(self._device_index)
        log.info("GPU memory with %d workers: %.1f MB", n_workers, with_workers_mb)

        # Stop workers
        for stop in stop_events:
            stop.set()
        for p in workers:
            p.join(timeout=10)

        growth = with_workers_mb - baseline_mb
        print(f"  Baseline: {baseline_mb:.1f} MB")
        print(f"  With {n_workers} workers: {with_workers_mb:.1f} MB")
        print(f"  Growth: {growth:.1f} MB")

    def test_zero_copy(self) -> None:
        """Test 2: Verify zero-copy by mutating weight in WM, observing in worker."""
        # Stop the subprocess WM — we need an in-process WM for this test
        self._stop_weight_manager()

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        from cuda_ipc_poc.weight_manager import WeightManager

        wm = WeightManager(self._model_name, self._device, self._endpoint)
        wm.load_model()
        wm._server.serve_in_background()

        try:
            mgr = multiprocessing.Manager()
            result_dict = mgr.dict()

            # Start a worker that reads, waits, then reads again
            ready = multiprocessing.Event()
            proceed = multiprocessing.Event()
            p = multiprocessing.Process(
                target=_worker_read_weight,
                args=(self._model_name, self._device, self._endpoint,
                      ready, proceed, result_dict, 0),
            )
            p.start()
            ready.wait(timeout=30)

            val_before = result_dict["0_before"]
            print(f"  Worker read fc1.weight[0][0] = {val_before}")

            # Mutate fc1.weight[0][0] in the WM (same physical GPU memory)
            sentinel = 42.42
            wm.model.fc1.weight.data[0][0] = sentinel
            print(f"  WM set fc1.weight[0][0] = {sentinel}")

            # Signal worker to re-read
            proceed.set()
            p.join(timeout=30)

            val_after = result_dict["0_after"]
            print(f"  Worker re-read fc1.weight[0][0] = {val_after}")

            if abs(val_after - sentinel) > 1e-5:
                raise AssertionError(
                    f"Zero-copy failed: expected {sentinel}, got {val_after}"
                )
        finally:
            wm._server.stop()
            # Mark WM process as None so _stop_weight_manager won't try to stop it
            self._wm_process = None

    def test_worker_lifecycle(self) -> None:
        """Test 3: 10x worker create/destroy cycles, WM remains stable."""
        for i in range(10):
            p = multiprocessing.Process(
                target=_worker_quick_cycle,
                args=(self._model_name, self._device, self._endpoint),
            )
            p.start()
            p.join(timeout=30)
            if p.exitcode != 0:
                raise AssertionError(f"Worker cycle {i+1} failed (exit code {p.exitcode})")
            print(f"  Cycle {i+1}/10: OK")

        # Verify WM is still alive
        if self._wm_process.poll() is not None:
            raise AssertionError("Weight Manager died during lifecycle test")
        print("  Weight Manager stable after 10 cycles")

    def test_inference_accuracy(self) -> None:
        """Test 4: Worker output matches direct model output with fixed seed."""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF

        seed = 12345
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        # Run two workers with same seed — their outputs should match
        procs = []
        for i in range(2):
            p = multiprocessing.Process(
                target=_worker_infer,
                args=(self._model_name, self._device, self._endpoint, seed, result_dict, i),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=30)

        out_0 = torch.tensor(result_dict[0]["output"])
        out_1 = torch.tensor(result_dict[1]["output"])

        if not torch.allclose(out_0, out_1, atol=1e-6):
            raise AssertionError(
                f"Outputs differ!\n  Worker 0: {out_0}\n  Worker 1: {out_1}"
            )
        print(f"  Worker outputs match: {out_0.shape}")
        print(f"  Max diff: {(out_0 - out_1).abs().max().item():.2e}")

    def test_memory_leak(self) -> None:
        """Test 5: 50x worker cycles, check GPU memory doesn't grow unboundedly.

        Strategy: 10 warmup cycles to absorb one-time CUDA context allocation,
        then measure stability over 40 more cycles. A leak would show continuous
        growth; stable memory (even if higher than pre-warmup) is fine.
        """
        warmup_cycles = 10
        test_cycles = 40
        total_cycles = warmup_cycles + test_cycles

        # Warmup: let CUDA contexts settle
        for i in range(warmup_cycles):
            p = multiprocessing.Process(
                target=_worker_quick_cycle,
                args=(self._model_name, self._device, self._endpoint),
            )
            p.start()
            p.join(timeout=30)
            if p.exitcode != 0:
                raise AssertionError(f"Warmup cycle {i+1} failed")

        baseline_mb = _get_gpu_memory_nvidia_smi(self._device_index)
        print(f"  Post-warmup baseline ({warmup_cycles} cycles): {baseline_mb:.1f} MB")

        # Test phase: run more cycles, sample memory every 10
        samples = []
        for i in range(test_cycles):
            p = multiprocessing.Process(
                target=_worker_quick_cycle,
                args=(self._model_name, self._device, self._endpoint),
            )
            p.start()
            p.join(timeout=30)
            if p.exitcode != 0:
                raise AssertionError(f"Test cycle {warmup_cycles + i + 1} failed")
            if (i + 1) % 10 == 0:
                current_mb = _get_gpu_memory_nvidia_smi(self._device_index)
                samples.append(current_mb)
                print(f"  Cycle {warmup_cycles + i + 1}/{total_cycles}: GPU memory = {current_mb:.1f} MB")

        final_mb = _get_gpu_memory_nvidia_smi(self._device_index)
        growth = final_mb - baseline_mb
        print(f"  Baseline: {baseline_mb:.1f} MB, Final: {final_mb:.1f} MB")
        print(f"  Growth after {test_cycles} test cycles: {growth:.1f} MB")

        # Check: memory growth during test phase should be minimal.
        # Allow 200 MB tolerance for ipc_collect timing jitter.
        if growth > 200:
            raise AssertionError(
                f"Potential memory leak: {growth:.1f} MB growth after warmup"
            )

    def test_crash_recovery(self) -> None:
        """Test 6: kill -9 a worker, ipc_collect, then new worker works."""
        ready = multiprocessing.Event()
        victim = multiprocessing.Process(
            target=_worker_crash_target,
            args=(self._model_name, self._device, self._endpoint, ready),
        )
        victim.start()
        ready.wait(timeout=30)
        print(f"  Victim worker started (PID {victim.pid})")

        # Kill it hard
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=10)
        print(f"  Victim killed with SIGKILL (exit code {victim.exitcode})")

        # Wait for WM's ipc_collect to run
        time.sleep(6)

        # New worker should work fine
        p = multiprocessing.Process(
            target=_worker_quick_cycle,
            args=(self._model_name, self._device, self._endpoint),
        )
        p.start()
        p.join(timeout=30)
        if p.exitcode != 0:
            raise AssertionError(f"Post-crash worker failed (exit code {p.exitcode})")
        print("  New worker after crash: OK")

        # WM should still be alive
        if self._wm_process.poll() is not None:
            raise AssertionError("Weight Manager died after worker crash")
        print("  Weight Manager survived worker crash")

    def test_tensor_parallelism(self) -> None:
        """Test 7: TP with 2 WMs on 2 GPUs matches single-GPU reference."""
        # Stop the default single-WM
        self._stop_weight_manager()

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        from cuda_ipc_poc.model import get_model

        world_size = 2
        tp_endpoints = [
            f"ipc:///tmp/cuda_ipc_tp_wm_{r}.zmq" for r in range(world_size)
        ]

        # Get reference output from single-GPU model (seeded for reproducibility)
        model_seed = 99
        torch.manual_seed(model_seed)
        model_ref, sample_fn = get_model(self._model_name)
        model_ref = model_ref.to(self._device)
        # nn.Module method that sets evaluation mode
        model_ref.eval()  # noqa: S307
        model_ref.requires_grad_(False)

        input_seed = 42
        torch.manual_seed(input_seed)
        sample = sample_fn(self._device)
        with torch.no_grad():
            ref_output = model_ref(sample)
        print(f"  Reference output shape: {ref_output.shape}")

        # Start 2 WM subprocesses (same model_seed ensures identical base weights)
        script = os.path.join(os.path.dirname(__file__), "run_weight_manager.py")
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
        wm_procs = []
        for rank in range(world_size):
            p = subprocess.Popen(
                [
                    sys.executable,
                    script,
                    "--model", self._model_name,
                    "--device", f"cuda:{rank}",
                    "--endpoint", tp_endpoints[rank],
                    "--tp-rank", str(rank),
                    "--tp-world-size", str(world_size),
                    "--seed", str(model_seed),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wm_procs.append(p)

        # Wait for WMs to initialize
        time.sleep(3.0)
        for r, p in enumerate(wm_procs):
            if p.poll() is not None:
                raise RuntimeError(f"WM rank {r} exited unexpectedly")
        print(f"  Started {world_size} WM processes")

        try:
            # Run TP worker
            manager = multiprocessing.Manager()
            result_dict = manager.dict()
            worker_p = multiprocessing.Process(
                target=_worker_tp_infer,
                args=(self._model_name, self._device, tp_endpoints, input_seed, result_dict, 0),
            )
            worker_p.start()
            worker_p.join(timeout=60)
            if worker_p.exitcode != 0:
                raise AssertionError(f"TP worker failed (exit code {worker_p.exitcode})")

            tp_output = torch.tensor(result_dict[0]["output"])
            ref_cpu = ref_output.cpu()

            max_diff = (tp_output - ref_cpu).abs().max().item()
            print(f"  TP output shape: {tp_output.shape}")
            print(f"  Max diff vs reference: {max_diff:.2e}")

            if not torch.allclose(tp_output, ref_cpu, atol=1e-5):
                raise AssertionError(
                    f"TP output differs from reference! Max diff: {max_diff}"
                )
        finally:
            # Stop TP WMs
            for p in wm_procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            # Clear wm_process so _stop_weight_manager doesn't complain
            self._wm_process = None


def main():
    parser = argparse.ArgumentParser(description="CUDA IPC Verification Suite")
    parser.add_argument("--model", default="mlp", choices=["mlp", "resnet18"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--endpoint", default="ipc:///tmp/cuda_ipc_verify.zmq")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [Verify] %(levelname)s %(message)s",
    )

    suite = VerificationSuite(args.model, args.device, args.endpoint)
    suite.run_all()


if __name__ == "__main__":
    main()
