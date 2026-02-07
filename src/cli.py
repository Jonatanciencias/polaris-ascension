#!/usr/bin/env python3
"""
Radeon RX 580 AI command-line interface.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .core.gpu import GPUManager
from .core.memory import MemoryManager
from .inference.base import InferenceConfig
from .inference.onnx_engine import ONNXInferenceEngine
from .optimization_engines.optimized_kernel_engine import OptimizedKernelEngine


class CLI:
    """CLI wrapper for system info, inference and GEMM benchmarking."""

    def __init__(self) -> None:
        self.gpu_manager = GPUManager()
        self.gpu_manager.initialize()
        self.memory_manager = MemoryManager()

    def info(self) -> int:
        """Display system and runtime capabilities."""
        print("\n" + "=" * 60)
        print("RADEON RX 580 AI - SYSTEM INFORMATION")
        print("=" * 60)

        gpu_info = self.gpu_manager.get_info()
        print("\nGPU Information:")
        if gpu_info:
            print(f"   Name: {gpu_info.name}")
            print(f"   Vendor: {gpu_info.vendor}")
            print(f"   Platform: {gpu_info.platform}")
            print(f"   Driver: {gpu_info.driver}")
            print(f"   OpenCL: {gpu_info.opencl_version}")
            print(f"   VRAM: {gpu_info.vram_gb:.1f} GB")
            print(f"   Backend: {self.gpu_manager.get_compute_backend().upper()}")
        else:
            print("   WARN No AMD OpenCL GPU detected")
            print("   Backend: CPU")

        stats = self.memory_manager.get_stats()
        print("\nMemory Information:")
        print(f"   RAM total: {stats.total_ram_gb:.1f} GB")
        print(f"   RAM available: {stats.available_ram_gb:.1f} GB")
        print(f"   RAM usage: {stats.ram_percent:.1f}%")
        print(f"   Strategy: {self.memory_manager.strategy.value}")

        print("\nCLI Capabilities:")
        print("   info: available")
        print("   classify: available (ONNX)")
        print("   benchmark: available (GEMM)")

        print("\n" + "=" * 60 + "\n")
        return 0

    def classify(
        self,
        image_paths: List[str],
        model_path: Optional[str] = None,
        fast: bool = False,
        ultra_fast: bool = False,
        batch_size: int = 1,
        top_k: int = 5,
        output: Optional[str] = None,
    ) -> int:
        """Run ONNX image classification."""
        if ultra_fast:
            precision = "int8"
            mode_name = "Ultra-Fast Mode (INT8)"
        elif fast:
            precision = "fp16"
            mode_name = "Fast Mode (FP16)"
        else:
            precision = "fp32"
            mode_name = "Standard Mode (FP32)"

        cfg = InferenceConfig(
            device="auto",
            precision=precision,
            batch_size=batch_size,
            enable_profiling=True,
            optimization_level=2,
        )
        engine = ONNXInferenceEngine(cfg, self.gpu_manager, self.memory_manager)

        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "examples/models/mobilenetv2.onnx")
        if not Path(model_path).exists():
            print(f"ERROR: model not found: {model_path}")
            print("Use --model to specify an ONNX file.")
            return 2

        print(f"\nInitializing {mode_name}...")
        print(f"Loading model from {model_path}...")
        engine.load_model(model_path)

        print(f"Processing {len(image_paths)} image(s)...")
        if len(image_paths) == 1:
            result = engine.infer(image_paths[0], top_k=top_k)
            self._print_result(image_paths[0], result, top_k)
        else:
            results = engine.infer_batch(image_paths, batch_size=batch_size)
            for img, result in zip(image_paths, results):
                self._print_result(img, result, top_k)

        if output:
            print(f"INFO: output export not implemented yet: {output}")

        return 0

    @staticmethod
    def _print_result(image_path: str, result: dict[str, Any], top_k: int) -> None:
        print(Path(image_path).name)
        print(f"   Top prediction: Class {result['top1_class']} ({result['top1_confidence']:.1%})")
        preds = result.get("predictions", [])[:max(1, top_k)]
        if len(preds) > 1:
            print(f"   Top {len(preds)} predictions:")
            for idx, pred in enumerate(preds, 1):
                print(f"      {idx}. Class {pred['class_id']}: {pred['confidence']:.1%}")

    def benchmark(self, size: int = 1024, iterations: int = 20) -> int:
        """Benchmark GEMM throughput with current OpenCL engine."""
        print("\n" + "=" * 60)
        print("GEMM PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Matrix size: {size}x{size}")
        print(f"Iterations: {iterations}")

        try:
            engine = OptimizedKernelEngine()
        except Exception as exc:
            print(f"ERROR: failed to initialize OpenCL engine: {exc}")
            return 2

        rng = np.random.default_rng(42)
        A = rng.standard_normal((size, size), dtype=np.float32)
        B = rng.standard_normal((size, size), dtype=np.float32)
        _ = engine.gemm(A, B)  # warm-up

        times: List[float] = []
        gflops_values: List[float] = []
        flops = 2.0 * (size**3)

        for _ in range(iterations):
            start = time.perf_counter()
            _ = engine.gemm(A, B)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            gflops_values.append(flops / elapsed / 1e9)

        mean_ms = float(np.mean(times) * 1000.0)
        p95_ms = float(np.percentile(times, 95) * 1000.0)
        mean_gflops = float(np.mean(gflops_values))
        peak_gflops = float(np.max(gflops_values))

        print("\nResults:")
        print(f"   Mean latency: {mean_ms:.2f} ms")
        print(f"   P95 latency: {p95_ms:.2f} ms")
        print(f"   Mean throughput: {mean_gflops:.1f} GFLOPS")
        print(f"   Peak throughput: {peak_gflops:.1f} GFLOPS")
        print("\n" + "=" * 60 + "\n")
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radeon RX 580 AI CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.cli info\n"
            "  python -m src.cli benchmark --size 1024 --iterations 20\n"
            "  python -m src.cli classify image.jpg --model model.onnx\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")
    sub.add_parser("info", help="Display system information")

    classify = sub.add_parser("classify", help="Classify image(s) with an ONNX model")
    classify.add_argument("images", nargs="+", help="Image file path(s)")
    classify.add_argument("--model", "-m", help="Path to ONNX model")
    classify.add_argument("--fast", "-f", action="store_true", help="Use FP16 mode")
    classify.add_argument("--ultra-fast", "-u", action="store_true", help="Use INT8 mode")
    classify.add_argument("--batch", "-b", type=int, default=1, help="Batch size")
    classify.add_argument("--top-k", "-k", type=int, default=5, help="Top-k predictions")
    classify.add_argument("--output", "-o", help="Output path")

    bench = sub.add_parser("benchmark", help="Run GEMM benchmark")
    bench.add_argument("--size", "-s", type=int, default=1024, help="Square matrix size")
    bench.add_argument("--iterations", "-i", type=int, default=20, help="Number of iterations")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        raise SystemExit(0)

    cli = CLI()
    if args.command == "info":
        raise SystemExit(cli.info())
    if args.command == "classify":
        raise SystemExit(
            cli.classify(
                image_paths=args.images,
                model_path=args.model,
                fast=args.fast,
                ultra_fast=args.ultra_fast,
                batch_size=args.batch,
                top_k=args.top_k,
                output=args.output,
            )
        )
    if args.command == "benchmark":
        raise SystemExit(cli.benchmark(size=args.size, iterations=args.iterations))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
