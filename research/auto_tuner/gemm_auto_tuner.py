#!/usr/bin/env python3
"""
GEMM Auto-Tuner Framework
=========================

Automated parameter search for optimal GEMM kernel configuration.

Search Space:
- Existing kernels: tile16, tile20, tile24
- Matrix sizes: Sweet spot region (1200-1800) + large sizes (2048-5120)
- Strategy: Exhaustive grid search

Expected runtime: 30-60 minutes GPU time
Expected outcome: Confirm tile20 @ 1400 or discover new sweet spot

Author: GEMM Optimization Project
Date: February 5, 2026
"""

import sys
import time
import json
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pyopencl as cl
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure pyopencl is installed: pip install pyopencl")
    sys.exit(1)


@dataclass
class TuningResult:
    """Single tuning experiment result"""

    tile_size: int
    matrix_size: int
    workgroup_x: int
    workgroup_y: int
    gflops: float
    avg_time_ms: float
    max_error: float
    timestamp: str
    runs: int


class GEMMAutoTuner:
    """
    Automated GEMM kernel parameter tuner.

    Searches parameter space systematically to find optimal configuration.
    Tests existing kernels (tile16, tile20, tile24) across matrix sizes.
    """

    def __init__(self, output_dir: str = "results/auto_tuner", verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Results tracking
        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None
        self.best_gflops: float = 0.0
        self._compiled_kernel_cache: Dict[Tuple[str, str, Tuple[str, ...]], cl.Kernel] = {}

        # Setup OpenCL
        self._setup_opencl()

        # Load kernels
        self.kernels = self._load_kernels()

        if self.verbose:
            print("=" * 70)
            print("GEMM AUTO-TUNER FRAMEWORK")
            print("=" * 70)
            print(f"Device: {self.device.name}")
            print(f"Compute Units: {self.device.max_compute_units}")
            print(f"Max Workgroup Size: {self.device.max_work_group_size}")
            print(f"Kernels loaded: {list(self.kernels.keys())}")
            print(f"Output directory: {self.output_dir}")
            print("=" * 70)

    def _setup_opencl(self):
        """Initialize OpenCL context"""
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        self.platform = platforms[0]
        devices = self.platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            raise RuntimeError("No GPU devices found")

        self.device = devices[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
        )

    def _load_kernels(self) -> Dict:
        """Load available GEMM kernels"""
        kernels = {}
        kernel_dir = project_root / "src" / "kernels"

        kernel_configs = {
            "tile20": {
                "path": kernel_dir / "gemm_tile20_production.cl",
                "kernel_name": "gemm_tile20_optimized",
                "tile_size": 20,
                "local_size": (10, 10),
            },
            "tile24": {
                "path": kernel_dir / "gemm_tile24_production.cl",
                "kernel_name": "gemm_tile24_vectorized",
                "tile_size": 24,
                "local_size": (12, 12),
            },
        }

        for name, config in kernel_configs.items():
            try:
                with open(config["path"], "r") as f:
                    kernel_source = f.read()

                program = cl.Program(self.ctx, kernel_source).build()
                kernel = getattr(program, config["kernel_name"])

                kernels[name] = {
                    "kernel": kernel,
                    "tile_size": config["tile_size"],
                    "local_size": config["local_size"],
                }

                if self.verbose:
                    print(
                        f"✓ Loaded {name}: tile_size={config['tile_size']}, "
                        f"local_size={config['local_size']}"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"✗ Could not load {name}: {e}")

        return kernels

    def define_search_space(self) -> Dict[str, List]:
        """
        Define parameter search space.

        Returns:
            Dictionary with parameter ranges
        """
        return {
            # Available kernels to test
            "kernels": list(self.kernels.keys()),
            # Matrix sizes - comprehensive sweep
            "matrix_sizes": [
                # Sweet spot region (fine-grained)
                1200,
                1250,
                1300,
                1350,
                1375,
                1400,
                1425,
                1450,
                1500,
                1550,
                1600,
                # Medium sizes
                1700,
                1800,
                1900,
                2000,
                # Large sizes
                2048,
                2560,
                3072,
                3584,
                4096,
                # Very large (if time permits)
                5120,
            ],
            # Number of runs per configuration
            "runs_per_config": 10,
            # Warmup runs
            "warmup_runs": 2,
        }

    def benchmark_kernel(
        self,
        kernel_name: str,
        M: int,
        N: int,
        K: int,
        runs: int = 10,
        warmup: int = 2,
        seed: Optional[int] = None,
        input_distribution: str = "standard_normal",
    ) -> Optional[TuningResult]:
        """
        Benchmark a specific kernel configuration.

        Args:
            kernel_name: Name of kernel ('tile16', 'tile20', 'tile24')
            M, N, K: Matrix dimensions
            runs: Number of benchmark runs
            warmup: Number of warmup runs
            seed: Optional deterministic seed
            input_distribution: Matrix value distribution ('standard_normal' | 'uniform')

        Returns:
            TuningResult or None if benchmark failed
        """
        if kernel_name not in self.kernels:
            return None

        kernel_info = self.kernels[kernel_name]
        return self._benchmark_with_kernel(
            kernel=kernel_info["kernel"],
            tile_size=int(kernel_info["tile_size"]),
            local_size=tuple(kernel_info["local_size"]),
            M=M,
            N=N,
            K=K,
            runs=runs,
            warmup=warmup,
            seed=seed,
            input_distribution=input_distribution,
        )

    def benchmark_custom_kernel(
        self,
        *,
        kernel_file: str,
        kernel_name: str,
        tile_size: int,
        local_size: Tuple[int, int],
        M: int,
        N: int,
        K: int,
        runs: int = 10,
        warmup: int = 2,
        seed: Optional[int] = None,
        input_distribution: str = "standard_normal",
        build_options: Optional[List[str]] = None,
    ) -> Optional[TuningResult]:
        """
        Benchmark a kernel defined by file/name with explicit execution parameters.
        """
        try:
            path = Path(kernel_file)
            if not path.is_absolute():
                path = (project_root / path).resolve()
            if not path.exists():
                if self.verbose:
                    print(f"  ❌ Kernel file not found: {path}")
                return None

            options = tuple(build_options or [])
            cache_key = (str(path), kernel_name, options)
            if cache_key not in self._compiled_kernel_cache:
                source = path.read_text()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        message=".*PyOpenCL compiler caching failed.*",
                    )
                    program = cl.Program(self.ctx, source).build(options=list(options))
                self._compiled_kernel_cache[cache_key] = getattr(program, kernel_name)

            kernel = self._compiled_kernel_cache[cache_key]
            return self._benchmark_with_kernel(
                kernel=kernel,
                tile_size=tile_size,
                local_size=local_size,
                M=M,
                N=N,
                K=K,
                runs=runs,
                warmup=warmup,
                seed=seed,
                input_distribution=input_distribution,
            )
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error benchmarking custom kernel {kernel_name}: {e}")
            return None

    def _benchmark_with_kernel(
        self,
        *,
        kernel: cl.Kernel,
        tile_size: int,
        local_size: Tuple[int, int],
        M: int,
        N: int,
        K: int,
        runs: int,
        warmup: int,
        seed: Optional[int],
        input_distribution: str,
    ) -> Optional[TuningResult]:
        A_buf = None
        B_buf = None
        C_buf = None
        try:
            if input_distribution not in {"standard_normal", "uniform"}:
                raise ValueError(
                    f"Unsupported input_distribution '{input_distribution}'. "
                    "Use 'standard_normal' or 'uniform'."
                )

            rng = np.random.default_rng(seed)
            if input_distribution == "standard_normal":
                A = rng.standard_normal((M, K), dtype=np.float32)
                B = rng.standard_normal((K, N), dtype=np.float32)
            else:
                A = rng.random((M, K), dtype=np.float32)
                B = rng.random((K, N), dtype=np.float32)
            C = np.zeros((M, N), dtype=np.float32)

            mf = cl.mem_flags
            A_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)

            global_size = (
                ((N + tile_size - 1) // tile_size) * int(local_size[0]),
                ((M + tile_size - 1) // tile_size) * int(local_size[1]),
            )

            for _ in range(warmup):
                event = kernel(
                    self.queue,
                    global_size,
                    local_size,
                    np.int32(M),
                    np.int32(N),
                    np.int32(K),
                    np.float32(1.0),
                    A_buf,
                    B_buf,
                    np.float32(0.0),
                    C_buf,
                )
                event.wait()

            times = []
            for _ in range(runs):
                event = kernel(
                    self.queue,
                    global_size,
                    local_size,
                    np.int32(M),
                    np.int32(N),
                    np.int32(K),
                    np.float32(1.0),
                    A_buf,
                    B_buf,
                    np.float32(0.0),
                    C_buf,
                )
                event.wait()
                elapsed = (event.profile.end - event.profile.start) * 1e-6
                times.append(elapsed)

            cl.enqueue_copy(self.queue, C, C_buf).wait()
            C_ref = A @ B
            max_error = float(np.max(np.abs(C - C_ref)))

            avg_time_ms = float(np.mean(times))
            gflops = (2.0 * M * N * K) / (avg_time_ms * 1e-3 * 1e9)

            result = TuningResult(
                tile_size=int(tile_size),
                matrix_size=M,
                workgroup_x=int(local_size[0]),
                workgroup_y=int(local_size[1]),
                gflops=float(gflops),
                avg_time_ms=avg_time_ms,
                max_error=max_error,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                runs=runs,
            )
            return result
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Error: {e}")
            return None
        finally:
            if A_buf is not None:
                A_buf.release()
            if B_buf is not None:
                B_buf.release()
            if C_buf is not None:
                C_buf.release()

    def run_search(self) -> Dict:
        """
        Execute automated parameter search.

        Returns:
            Summary dictionary with results
        """
        search_space = self.define_search_space()

        kernels = search_space["kernels"]
        matrix_sizes = search_space["matrix_sizes"]
        runs = search_space["runs_per_config"]
        warmup = search_space["warmup_runs"]

        total_configs = len(kernels) * len(matrix_sizes)
        completed = 0

        print(f"\n🔍 Starting parameter search...")
        print(f"   Kernels: {len(kernels)} ({', '.join(kernels)})")
        print(f"   Matrix sizes: {len(matrix_sizes)}")
        print(f"   Total configurations: {total_configs}")
        print(f"   Runs per config: {runs}")
        print(f"   Warmup runs: {warmup}")
        print()

        start_time = time.time()

        # Search through parameter space
        for kernel_idx, kernel_name in enumerate(kernels):
            print(f"\n{'='*70}")
            print(f"Testing {kernel_name} ({kernel_idx+1}/{len(kernels)})")
            print(f"{'='*70}")

            kernel_results = []

            for matrix_size in matrix_sizes:
                completed += 1
                percent = (completed / total_configs) * 100

                print(f"\n[{completed}/{total_configs} - {percent:.1f}%] ", end="")
                print(f"{kernel_name} @ {matrix_size}×{matrix_size}... ", end="", flush=True)

                result = self.benchmark_kernel(
                    kernel_name=kernel_name,
                    M=matrix_size,
                    N=matrix_size,
                    K=matrix_size,
                    runs=runs,
                    warmup=warmup,
                )

                if result is None:
                    print("❌ FAILED")
                    continue

                # Check correctness
                if result.max_error > 0.01:
                    print(f"❌ ERROR TOO HIGH (error: {result.max_error:.6f})")
                    continue

                # Update tracking
                self.results.append(result)
                kernel_results.append(result)

                # Check if new best
                if result.gflops > self.best_gflops:
                    self.best_gflops = result.gflops
                    self.best_result = result
                    print(f"🏆 {result.gflops:.1f} GFLOPS (NEW BEST!)")
                else:
                    print(f"✅ {result.gflops:.1f} GFLOPS")

                # Save intermediate results
                if completed % 5 == 0:  # Save every 5 configs
                    self._save_results()

            # Kernel summary
            if kernel_results:
                best_kernel = max(kernel_results, key=lambda r: r.gflops)
                worst_kernel = min(kernel_results, key=lambda r: r.gflops)
                avg_gflops = np.mean([r.gflops for r in kernel_results])

                print(f"\n📊 {kernel_name} summary:")
                print(
                    f"   Best:    {best_kernel.gflops:.1f} GFLOPS @ {best_kernel.matrix_size}×{best_kernel.matrix_size}"
                )
                print(
                    f"   Worst:   {worst_kernel.gflops:.1f} GFLOPS @ {worst_kernel.matrix_size}×{worst_kernel.matrix_size}"
                )
                print(f"   Average: {avg_gflops:.1f} GFLOPS")
                print(f"   Configs: {len(kernel_results)}/{len(matrix_sizes)}")
            else:
                print(f"\n⚠️  No valid results for {kernel_name}")

        # Search complete
        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print("SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"✅ Configurations tested: {len(self.results)}/{total_configs}")
        print(f"🏆 Best configuration:")
        if self.best_result:
            print(
                f"   tile{self.best_result.tile_size} @ {self.best_result.matrix_size}×{self.best_result.matrix_size}"
            )
            print(f"   Performance: {self.best_result.gflops:.1f} GFLOPS")
            print(f"   Workgroup: ({self.best_result.workgroup_x}, {self.best_result.workgroup_y})")
            print(f"   Average time: {self.best_result.avg_time_ms:.2f} ms")
            print(f"   Max error: {self.best_result.max_error:.6f}")

        # Generate summary
        summary = self._generate_summary(elapsed)

        # Save final results
        self._save_results()
        self._save_summary(summary)

        return summary

    def _save_results(self):
        """Save results to CSV"""
        if not self.results:
            return

        csv_path = self.output_dir / "tuning_results.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

    def _generate_summary(self, elapsed_time: float) -> Dict:
        """Generate summary statistics"""
        if not self.results:
            return {"error": "No results"}

        # Top 10 configurations
        top_10 = sorted(self.results, key=lambda r: r.gflops, reverse=True)[:10]

        # Per-kernel statistics
        kernel_stats = {}
        for result in self.results:
            tile = result.tile_size
            if tile not in kernel_stats:
                kernel_stats[tile] = []
            kernel_stats[tile].append(result.gflops)

        kernel_summary = {
            f"tile{tile}": {
                "max_gflops": max(gflops),
                "mean_gflops": np.mean(gflops),
                "min_gflops": min(gflops),
                "std_gflops": np.std(gflops),
                "count": len(gflops),
            }
            for tile, gflops in kernel_stats.items()
        }

        return {
            "elapsed_time_minutes": elapsed_time / 60,
            "total_configurations": len(self.results),
            "best_configuration": asdict(self.best_result) if self.best_result else None,
            "top_10_configurations": [asdict(r) for r in top_10],
            "kernel_summary": kernel_summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save_summary(self, summary: Dict):
        """Save summary to JSON"""
        json_path = self.output_dir / "tuning_summary.json"

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"   CSV: {self.output_dir / 'tuning_results.csv'}")
        print(f"   JSON: {json_path}")


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("GEMM AUTO-TUNER - Automated Parameter Search")
    print("=" * 70)
    print("\nThis will systematically search for optimal GEMM configuration.")
    print("Expected runtime: 2-4 hours")
    print("\nPress Ctrl+C to stop early (results will be saved)")
    print("=" * 70)

    # Create tuner
    tuner = GEMMAutoTuner(output_dir="results/auto_tuner", verbose=True)

    try:
        # Run search
        summary = tuner.run_search()

        # Print final report
        print("\n" + "=" * 70)
        print("AUTO-TUNING COMPLETE")
        print("=" * 70)

        if summary.get("best_configuration"):
            best = summary["best_configuration"]
            print(f"\n🏆 OPTIMAL CONFIGURATION FOUND:")
            print(f"   Tile size: {best['tile_size']}")
            print(f"   Matrix size: {best['matrix_size']}×{best['matrix_size']}")
            print(f"   Performance: {best['gflops']:.1f} GFLOPS")
            print(f"   Workgroup: ({best['workgroup_x']}, {best['workgroup_y']})")

        print(f"\n✅ Results saved to: results/auto_tuner/")
        print("   - tuning_results.csv (all configurations)")
        print("   - tuning_summary.json (summary + top 10)")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user")
        print(f"✅ Partial results saved ({len(tuner.results)} configurations)")
        return 1
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
