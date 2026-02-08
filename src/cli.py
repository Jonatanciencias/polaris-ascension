#!/usr/bin/env python3
"""
Radeon RX 580 AI command-line interface.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .benchmarking.reporting import (
    markdown_table,
    report_paths,
    save_json_report,
    save_markdown_report,
)
from .core.gpu import GPUManager
from .core.memory import MemoryManager
from .inference.base import InferenceConfig
from .inference.onnx_engine import ONNXInferenceEngine


class CLI:
    """CLI wrapper for system info, inference and GEMM benchmarking."""

    def __init__(self, *, initialize_runtime: bool = True) -> None:
        self.gpu_manager: GPUManager | None = None
        self.memory_manager: MemoryManager | None = None
        if initialize_runtime:
            self.gpu_manager = GPUManager()
            self.gpu_manager.initialize()
            self.memory_manager = MemoryManager()

    def info(self) -> int:
        """Display system and runtime capabilities."""
        if self.gpu_manager is None or self.memory_manager is None:
            self.gpu_manager = GPUManager()
            self.gpu_manager.initialize()
            self.memory_manager = MemoryManager()

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
        if self.gpu_manager is None or self.memory_manager is None:
            self.gpu_manager = GPUManager()
            self.gpu_manager.initialize()
            self.memory_manager = MemoryManager()

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

    def _persist_benchmark_report(
        self,
        *,
        prefix: str,
        report: dict[str, Any],
        markdown: str,
        report_dir: str,
    ) -> None:
        json_path, md_path = report_paths(prefix=prefix, output_dir=report_dir)
        save_json_report(json_path, report)
        save_markdown_report(md_path, markdown)
        print(f"\nReport JSON: {json_path}")
        print(f"Report MD:   {md_path}")

    def benchmark(
        self,
        size: int = 1024,
        iterations: int = 20,
        mode: str = "engine",
        kernel: str = "auto",
        sessions: int = 5,
        opencl_platform: str | None = None,
        opencl_device: str | None = None,
        rusticl_enable: str | None = None,
        report_dir: str = "results/benchmark_reports",
        no_report: bool = False,
    ) -> int:
        """Benchmark GEMM throughput using engine or production kernels."""
        print("\n" + "=" * 60)
        print("GEMM PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"Matrix size: {size}x{size}")
        print(f"Mode: {mode}")
        print(f"Iterations/session: {iterations}")
        print(f"Sessions: {sessions}")
        if mode == "production":
            print(f"Kernel mode: {kernel}")
            if opencl_platform is not None:
                print(f"OpenCL platform selector: {opencl_platform}")
            if opencl_device is not None:
                print(f"OpenCL device selector: {opencl_device}")
            if rusticl_enable is not None:
                print(f"RUSTICL_ENABLE: {rusticl_enable}")

        if mode == "engine":
            try:
                from .optimization_engines.optimized_kernel_engine import OptimizedKernelEngine

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

            report = {
                "metadata": {
                    "mode": "engine",
                    "size": size,
                    "iterations": iterations,
                    "seed": 42,
                },
                "summary": {
                    "mean_latency_ms": mean_ms,
                    "p95_latency_ms": p95_ms,
                    "mean_gflops": mean_gflops,
                    "peak_gflops": peak_gflops,
                },
            }
            md = (
                "# CLI Benchmark Report (Engine)\n\n"
                + markdown_table(
                    headers=["Metric", "Value"],
                    rows=[
                        ("Size", f"{size}x{size}"),
                        ("Iterations", iterations),
                        ("Mean latency (ms)", f"{mean_ms:.2f}"),
                        ("P95 latency (ms)", f"{p95_ms:.2f}"),
                        ("Mean throughput (GFLOPS)", f"{mean_gflops:.1f}"),
                        ("Peak throughput (GFLOPS)", f"{peak_gflops:.1f}"),
                    ],
                )
                + "\n"
            )
            if not no_report:
                self._persist_benchmark_report(
                    prefix="cli_engine_benchmark",
                    report=report,
                    markdown=md,
                    report_dir=report_dir,
                )
            print("\n" + "=" * 60 + "\n")
            return 0

        if mode == "production":
            try:
                if rusticl_enable is not None:
                    os.environ["RUSTICL_ENABLE"] = str(rusticl_enable)
                # Lazy import so environment selectors (e.g. RUSTICL_ENABLE)
                # can be set before PyOpenCL platform discovery.
                from .benchmarking.production_kernel_benchmark import run_production_benchmark

                report = run_production_benchmark(
                    size=size,
                    iterations=iterations,
                    sessions=sessions,
                    kernel=kernel,
                    seed=42,
                    opencl_platform=opencl_platform,
                    opencl_device=opencl_device,
                    rusticl_enable=rusticl_enable,
                )
            except Exception as exc:
                print(f"ERROR: production benchmark failed: {exc}")
                return 2

            summary = report["summary"]
            peak = summary["peak_gflops"]
            avg = summary["avg_gflops"]
            time_stats = summary["time_ms"]
            err = summary["max_error"]

            print("\nResults:")
            print(
                "   Peak throughput mean: "
                f"{peak['mean']:.1f} GFLOPS [{peak['min']:.1f}, {peak['max']:.1f}]"
            )
            print(f"   Avg throughput mean:  {avg['mean']:.1f} GFLOPS")
            print(
                f"   Kernel time mean:     {time_stats['mean']:.3f} ms "
                f"(p95 {time_stats['p95']:.3f})"
            )
            print(f"   Max error mean:       {err['mean']:.6f}")

            md = (
                "# CLI Benchmark Report (Production)\n\n"
                + markdown_table(
                    headers=["Metric", "Value"],
                    rows=[
                        ("Size", f"{size}x{size}"),
                        ("Kernel mode", kernel),
                        ("OpenCL platform selector", opencl_platform or "auto/env"),
                        ("OpenCL device selector", opencl_device or "auto/env"),
                        ("RUSTICL_ENABLE", rusticl_enable or "<unchanged>"),
                        ("Sessions", sessions),
                        ("Iterations/session", iterations),
                        (
                            "Peak throughput mean (GFLOPS)",
                            f"{peak['mean']:.1f} [{peak['min']:.1f}, {peak['max']:.1f}]",
                        ),
                        ("Avg throughput mean (GFLOPS)", f"{avg['mean']:.1f}"),
                        ("Kernel time mean (ms)", f"{time_stats['mean']:.3f}"),
                        ("Kernel time p95 (ms)", f"{time_stats['p95']:.3f}"),
                        ("Max error mean", f"{err['mean']:.6f}"),
                    ],
                )
                + "\n"
            )
            if not no_report:
                self._persist_benchmark_report(
                    prefix="cli_production_benchmark",
                    report=report,
                    markdown=md,
                    report_dir=report_dir,
                )
            print("\n" + "=" * 60 + "\n")
            return 0

        print(f"ERROR: unsupported benchmark mode '{mode}'")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radeon RX 580 AI CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.cli info\n"
            "  python -m src.cli benchmark --size 1024 --iterations 20\n"
            "  python -m src.cli benchmark --mode production --kernel auto --size 1400 --sessions 10 --iterations 20\n"
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
    bench.add_argument(
        "--mode",
        choices=["engine", "production"],
        default="engine",
        help="Benchmark mode: generic engine or production tile kernels",
    )
    bench.add_argument(
        "--kernel",
        choices=[
            "auto",
            "auto_t3_controlled",
            "auto_t5_guarded",
            "tile20",
            "tile20_v3_1400",
            "tile24",
        ],
        default="auto",
        help="Kernel mode for --mode production",
    )
    bench.add_argument(
        "--sessions",
        type=int,
        default=5,
        help="Repeated sessions for aggregate metrics (production mode)",
    )
    bench.add_argument(
        "--opencl-platform",
        default=None,
        help="Explicit OpenCL platform selector (name substring).",
    )
    bench.add_argument(
        "--opencl-device",
        default=None,
        help="Explicit OpenCL device selector (name substring).",
    )
    bench.add_argument(
        "--rusticl-enable",
        nargs="?",
        const="radeonsi",
        default=None,
        help="Set RUSTICL_ENABLE before OpenCL discovery (default value: radeonsi).",
    )
    bench.add_argument(
        "--report-dir",
        default="results/benchmark_reports",
        help="Directory for auto-saved benchmark reports",
    )
    bench.add_argument(
        "--no-report",
        action="store_true",
        help="Do not persist benchmark report artifacts",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        raise SystemExit(0)

    if args.command == "info":
        cli = CLI(initialize_runtime=True)
        raise SystemExit(cli.info())
    if args.command == "classify":
        cli = CLI(initialize_runtime=True)
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
        # Delay GPU/OpenCL runtime initialization so selector env vars
        # (e.g. RUSTICL_ENABLE) can be applied before platform discovery.
        cli = CLI(initialize_runtime=False)
        raise SystemExit(
            cli.benchmark(
                size=args.size,
                iterations=args.iterations,
                mode=args.mode,
                kernel=args.kernel,
                sessions=args.sessions,
                opencl_platform=args.opencl_platform,
                opencl_device=args.opencl_device,
                rusticl_enable=args.rusticl_enable,
                report_dir=args.report_dir,
                no_report=args.no_report,
            )
        )

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
