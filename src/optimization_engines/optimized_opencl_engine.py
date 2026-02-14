#!/usr/bin/env python3
"""
🚀 HIGHLY OPTIMIZED OPENCL ENGINE FOR RADEON RX 580
==================================================

Professional OpenCL implementation with advanced optimizations:
- Vectorized kernels (float4/float8)
- Shared memory tiling (32x32 tiles)
- Memory coalescing
- Work-group optimizations
- Multiple accumulators for latency hiding
- Loop unrolling
- Register blocking

Target: AMD Radeon RX 580 (Polaris 10)
Architecture: GCN 4.0, 36 compute units, 8GB GDDR5
Goal: Achieve 1000+ GFLOPS sustained performance

Author: AI Assistant
Date: 2026-01-25
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyopencl as cl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OpenCLOptimizationConfig:
    """Configuration for OpenCL optimizations"""

    tile_size: int = 32
    vector_size: int = 4
    work_per_thread: int = 8
    local_work_size: Tuple[int, int] = (16, 16)
    use_shared_memory: bool = True
    use_vectorization: bool = True
    unroll_factor: int = 8


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""

    gflops: float
    bandwidth_gb_s: float
    kernel_time_ms: float
    total_time_ms: float
    efficiency_percent: float
    theoretical_peak_gflops: float = 6170.0  # Radeon RX 580 theoretical peak


class OptimizedOpenCLEngine:
    """
    Highly optimized OpenCL engine for Radeon RX 580
    Implements advanced GPU computing techniques for maximum performance
    """

    def __init__(self, config: Optional[OpenCLOptimizationConfig] = None):
        self.config = config or OpenCLOptimizationConfig()

        # Initialize OpenCL
        self.platform: Optional[cl.Platform] = None
        self.device: Optional[cl.Device] = None
        self.context: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.program: Optional[cl.Program] = None

        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []

        # Initialize OpenCL environment
        self._initialize_opencl()

        # Load optimized kernels
        self._load_kernels()

        logger.info("🚀 Optimized OpenCL Engine initialized for Radeon RX 580")

    def _initialize_opencl(self):
        """Initialize OpenCL environment with optimal settings"""
        try:
            # Get AMD platform
            platforms = cl.get_platforms()
            amd_platforms = [
                p for p in platforms if "AMD" in p.name or "Advanced Micro Devices" in p.name
            ]
            self.platform = amd_platforms[0] if amd_platforms else platforms[0]
            platform = self.platform

            # Get Radeon RX 580 device
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            radeon_devices = [d for d in devices if "Radeon RX 580" in d.name or "580" in d.name]
            self.device = radeon_devices[0] if radeon_devices else devices[0]
            device = self.device

            logger.info(f"Selected device: {device.name}")
            logger.info(f"Compute units: {device.max_compute_units}")
            logger.info(f"Max work group size: {device.max_work_group_size}")
            logger.info(f"Local memory size: {device.local_mem_size // 1024} KB")

            # Create context and queue with optimal settings
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(
                self.context, device, properties=cl.command_queue_properties.PROFILING_ENABLE
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            raise

    def _load_kernels(self):
        """Load optimized OpenCL kernels"""
        try:
            with open("optimized_kernels.cl", "r") as f:
                kernel_source = f.read()

            # Build program with optimizations
            build_options = [
                "-cl-mad-enable",  # Enable fused multiply-add
                "-cl-no-signed-zeros",  # Assume no signed zeros
                "-cl-unsafe-math-optimizations",  # Enable unsafe math optimizations
                "-cl-finite-math-only",  # Assume finite math only
                "-cl-fast-relaxed-math",  # Fast relaxed math
                f"-DTILE_SIZE={self.config.tile_size}",
                f"-DVECTOR_SIZE={self.config.vector_size}",
                f"-DWORK_PER_THREAD={self.config.work_per_thread}",
            ]

            context = self.context
            if context is None:
                raise RuntimeError("OpenCL context not initialized")

            self.program = cl.Program(context, kernel_source).build(options=build_options)
            logger.info("✅ Optimized kernels loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load kernels: {e}")
            raise

    def _get_optimal_work_groups(self, M: int, N: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate optimal work group configuration"""
        # Use tile-based work groups for better cache utilization
        global_work_size = (
            (N + self.config.tile_size - 1) // self.config.tile_size * self.config.tile_size,
            (M + self.config.tile_size - 1) // self.config.tile_size * self.config.tile_size,
        )

        local_work_size = self.config.local_work_size

        return global_work_size, local_work_size

    def optimized_gemm(
        self, A: np.ndarray, B: np.ndarray, alpha: float = 1.0, beta: float = 0.0
    ) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Perform highly optimized matrix multiplication using vectorized tiled kernels

        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)
            alpha: Scaling factor for A*B
            beta: Scaling factor for C

        Returns:
            Result matrix C and performance metrics
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions don't match"

        C = np.zeros((M, N), dtype=np.float32)
        context = self.context
        queue = self.queue
        program = self.program
        device = self.device
        if context is None or queue is None or program is None or device is None:
            raise RuntimeError("OpenCL runtime not initialized")

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
        C_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

        # Get optimal work group configuration
        global_work_size, local_work_size = self._get_optimal_work_groups(M, N)

        # Select kernel based on configuration
        kernel_args: tuple[Any, ...]
        if self.config.use_vectorization and N % self.config.vector_size == 0:
            kernel = program.gemm_vectorized_tiled
            logger.info("Using vectorized tiled GEMM kernel")
            kernel_args = (
                np.int32(M),
                np.int32(N),
                np.int32(K),
                np.float32(alpha),
                A_buf,
                B_buf,
                np.float32(beta),
                C_buf,
            )
        else:
            kernel = program.gemm_shared_memory_tiled
            logger.info("Using shared memory tiled GEMM kernel")
            kernel_args = (np.int32(M), np.int32(N), np.int32(K), A_buf, B_buf, C_buf)

        # Set kernel arguments
        kernel.set_args(*kernel_args)

        # Execute kernel with timing
        start_time = time.time()
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
        event.wait()
        kernel_time = (event.profile.end - event.profile.start) * 1e-9  # Convert to seconds

        # Read result
        cl.enqueue_copy(queue, C, C_buf)

        total_time = time.time() - start_time

        # Calculate performance metrics
        operations = 2 * M * N * K  # Multiply-add operations
        gflops = operations / (kernel_time * 1e9)
        bandwidth = (A.nbytes + B.nbytes + C.nbytes) / (kernel_time * 1e9) / (1024**3)  # GB/s
        efficiency = (gflops / device.max_compute_units / 100) * 100  # Rough efficiency estimate

        metrics = PerformanceMetrics(
            gflops=gflops,
            bandwidth_gb_s=bandwidth,
            kernel_time_ms=kernel_time * 1000,
            total_time_ms=total_time * 1000,
            efficiency_percent=efficiency,
        )

        self.metrics_history.append(metrics)

        logger.info(".2f" ".2f" ".2f" ".1f")

        return C, metrics

    def optimized_cw_gemm(
        self, A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Perform Coppersmith-Winograd optimized matrix multiplication

        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)

        Returns:
            Result matrix C and performance metrics
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions don't match"

        C = np.zeros((M, N), dtype=np.float32)
        context = self.context
        queue = self.queue
        program = self.program
        device = self.device
        if context is None or queue is None or program is None or device is None:
            raise RuntimeError("OpenCL runtime not initialized")

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
        C_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, size=C.nbytes)

        # Work group configuration
        global_work_size = (M, N)
        local_work_size = None  # Let OpenCL decide

        kernel = program.cw_matrix_multiply_optimized
        kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

        # Execute kernel
        start_time = time.time()
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
        event.wait()
        kernel_time = (event.profile.end - event.profile.start) * 1e-9

        cl.enqueue_copy(queue, C, C_buf)
        total_time = time.time() - start_time

        # Performance metrics
        operations = 2 * M * N * K
        gflops = operations / (kernel_time * 1e9)

        metrics = PerformanceMetrics(
            gflops=gflops,
            bandwidth_gb_s=(A.nbytes + B.nbytes + C.nbytes) / (kernel_time * 1e9) / (1024**3),
            kernel_time_ms=kernel_time * 1000,
            total_time_ms=total_time * 1000,
            efficiency_percent=(gflops / device.max_compute_units / 100) * 100,
        )

        logger.info(".2f")

        return C, metrics

    def optimized_low_rank_gemm(
        self, A_approx: np.ndarray, B_approx: np.ndarray
    ) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Perform low-rank optimized matrix multiplication

        Args:
            A_approx: Low-rank approximation of A (M x R)
            B_approx: Low-rank approximation of B (R x N)

        Returns:
            Result matrix C and performance metrics
        """
        M, R = A_approx.shape
        R2, N = B_approx.shape
        assert R == R2, "Rank dimensions don't match"

        C = np.zeros((M, N), dtype=np.float32)
        context = self.context
        queue = self.queue
        program = self.program
        device = self.device
        if context is None or queue is None or program is None or device is None:
            raise RuntimeError("OpenCL runtime not initialized")

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_approx.astype(np.float32)
        )
        B_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_approx.astype(np.float32)
        )
        C_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, size=C.nbytes)

        # Work group configuration
        global_work_size = (M, N)
        local_work_size = None

        kernel = program.low_rank_gemm_optimized
        kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(R))

        # Execute kernel
        start_time = time.time()
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
        event.wait()
        kernel_time = (event.profile.end - event.profile.start) * 1e-9

        cl.enqueue_copy(queue, C, C_buf)
        total_time = time.time() - start_time

        # Performance metrics
        operations = 2 * M * N * R  # Reduced operations due to low rank
        gflops = operations / (kernel_time * 1e9)

        metrics = PerformanceMetrics(
            gflops=gflops,
            bandwidth_gb_s=(A_approx.nbytes + B_approx.nbytes + C.nbytes)
            / (kernel_time * 1e9)
            / (1024**3),
            kernel_time_ms=kernel_time * 1000,
            total_time_ms=total_time * 1000,
            efficiency_percent=(gflops / device.max_compute_units / 100) * 100,
        )

        logger.info(".2f")

        return C, metrics

    def optimized_gemm_ultra(
        self, A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Ultra-high performance GEMM using the most optimized kernel targeting 1000+ GFLOPS

        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)

        Returns:
            Result matrix C and performance metrics
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Matrix dimensions don't match"

        C = np.zeros((M, N), dtype=np.float32)
        context = self.context
        queue = self.queue
        program = self.program
        device = self.device
        if context is None or queue is None or program is None or device is None:
            raise RuntimeError("OpenCL runtime not initialized")

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
        B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
        C_buf = cl.Buffer(context, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, size=C.nbytes)

        # Ultra-optimized work group configuration
        TS = 32  # Tile size
        global_work_size = ((N + TS - 1) // TS * TS, (M + TS - 1) // TS * TS)
        local_work_size = (16, 16)  # Optimized for Radeon RX 580

        kernel = program.gemm_ultra_optimized
        kernel.set_args(np.int32(M), np.int32(N), np.int32(K), A_buf, B_buf, C_buf)

        # Execute kernel with timing
        start_time = time.time()
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)
        event.wait()
        kernel_time = (event.profile.end - event.profile.start) * 1e-9

        # Read result
        cl.enqueue_copy(queue, C, C_buf)
        total_time = time.time() - start_time

        # Performance metrics
        operations = 2 * M * N * K
        gflops = operations / (kernel_time * 1e9)
        bandwidth = (A.nbytes + B.nbytes + C.nbytes) / (kernel_time * 1e9) / (1024**3)

        metrics = PerformanceMetrics(
            gflops=gflops,
            bandwidth_gb_s=bandwidth,
            kernel_time_ms=kernel_time * 1000,
            total_time_ms=total_time * 1000,
            efficiency_percent=(gflops / device.max_compute_units / 100) * 100,
        )

        logger.info(".2f" ".2f" ".2f" ".1f")

        return C, metrics

    def benchmark_optimization(self, sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Comprehensive benchmark of optimization techniques

        Args:
            sizes: List of matrix sizes to test

        Returns:
            Benchmark results dictionary
        """
        if sizes is None:
            sizes = [512, 1024, 2048, 4096]

        results: Dict[str, Any] = {
            "sizes": sizes,
            "vectorized_gemm": [],
            "shared_memory_gemm": [],
            "ultra_optimized_gemm": [],
            "cw_gemm": [],
            "low_rank_gemm": [],
        }

        logger.info("🚀 Starting comprehensive OpenCL optimization benchmark")

        for size in sizes:
            logger.info(f"Benchmarking size {size}x{size}")

            # Generate test matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Benchmark vectorized GEMM
            try:
                _, metrics = self.optimized_gemm(A, B)
                results["vectorized_gemm"].append(metrics.gflops)
            except Exception as e:
                logger.error(f"Vectorized GEMM failed for size {size}: {e}")
                results["vectorized_gemm"].append(0.0)

            # Benchmark shared memory GEMM
            self.config.use_vectorization = False
            try:
                _, metrics = self.optimized_gemm(A, B)
                results["shared_memory_gemm"].append(metrics.gflops)
            except Exception as e:
                logger.error(f"Shared memory GEMM failed for size {size}: {e}")
                results["shared_memory_gemm"].append(0.0)
            self.config.use_vectorization = True

            # Benchmark ultra-optimized GEMM
            try:
                _, metrics = self.optimized_gemm_ultra(A, B)
                results["ultra_optimized_gemm"].append(metrics.gflops)
            except Exception as e:
                logger.error(f"Ultra-optimized GEMM failed for size {size}: {e}")
                results["ultra_optimized_gemm"].append(0.0)

            # Benchmark CW GEMM (for smaller sizes)
            if size <= 1024:
                try:
                    _, metrics = self.optimized_cw_gemm(A, B)
                    results["cw_gemm"].append(metrics.gflops)
                except Exception as e:
                    logger.error(f"CW GEMM failed for size {size}: {e}")
                    results["cw_gemm"].append(0.0)
            else:
                results["cw_gemm"].append(0.0)  # CW not suitable for large matrices

            # Benchmark low-rank GEMM
            rank = min(size // 4, 256)  # Adaptive rank
            A_approx = np.random.randn(size, rank).astype(np.float32)
            B_approx = np.random.randn(rank, size).astype(np.float32)
            try:
                _, metrics = self.optimized_low_rank_gemm(A_approx, B_approx)
                results["low_rank_gemm"].append(metrics.gflops)
            except Exception as e:
                logger.error(f"Low-rank GEMM failed for size {size}: {e}")
                results["low_rank_gemm"].append(0.0)

        # Find best results
        best_results = {}
        for method in [
            "vectorized_gemm",
            "shared_memory_gemm",
            "ultra_optimized_gemm",
            "cw_gemm",
            "low_rank_gemm",
        ]:
            performances = results[method]
            if performances:
                max_perf = max(performances)
                best_size_idx = performances.index(max_perf)
                best_results[method] = {"peak_gflops": max_perf, "best_size": sizes[best_size_idx]}

        results["best_results"] = best_results

        # Save benchmark results
        np.savez(
            "opencl_optimization_benchmark.npz",
            sizes=np.array(sizes, dtype=np.int32),
            vectorized_gemm=np.array(results["vectorized_gemm"], dtype=np.float32),
            shared_memory_gemm=np.array(results["shared_memory_gemm"], dtype=np.float32),
            ultra_optimized_gemm=np.array(results["ultra_optimized_gemm"], dtype=np.float32),
            cw_gemm=np.array(results["cw_gemm"], dtype=np.float32),
            low_rank_gemm=np.array(results["low_rank_gemm"], dtype=np.float32),
        )

        logger.info("✅ Benchmark completed")
        logger.info(
            f"Best vectorized GEMM: {best_results.get('vectorized_gemm', {}).get('peak_gflops', 0):.2f} GFLOPS"
        )
        logger.info(
            f"Best shared memory GEMM: {best_results.get('shared_memory_gemm', {}).get('peak_gflops', 0):.2f} GFLOPS"
        )
        logger.info(
            f"Best ultra-optimized GEMM: {best_results.get('ultra_optimized_gemm', {}).get('peak_gflops', 0):.2f} GFLOPS"
        )

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {}

        gflops_values = [m.gflops for m in self.metrics_history]
        kernel_times = [m.kernel_time_ms for m in self.metrics_history]
        efficiencies = [m.efficiency_percent for m in self.metrics_history]

        return {
            "total_operations": len(self.metrics_history),
            "average_gflops": np.mean(gflops_values),
            "peak_gflops": np.max(gflops_values),
            "average_kernel_time_ms": np.mean(kernel_times),
            "min_kernel_time_ms": np.min(kernel_times),
            "average_efficiency_percent": np.mean(efficiencies),
            "max_efficiency_percent": np.max(efficiencies),
        }


def main():
    """Main function for testing the optimized OpenCL engine"""
    logger.info("🚀 Starting Optimized OpenCL Engine Test")

    # Initialize engine
    engine = OptimizedOpenCLEngine()

    # Test with different sizes
    test_sizes = [512, 1024, 2048]

    for size in test_sizes:
        logger.info(f"Testing {size}x{size} matrix multiplication")

        # Generate test matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Test optimized GEMM
        C_gpu, metrics = engine.optimized_gemm(A, B)

        # Verify correctness (compare with NumPy)
        C_cpu = np.dot(A, B)
        max_error = np.max(np.abs(C_gpu - C_cpu))
        logger.info(f"Max error vs NumPy: {max_error:.2e}")

        if max_error > 1e-3:
            logger.warning("Large error detected - possible kernel bug")

    # Run comprehensive benchmark
    logger.info("Running comprehensive benchmark...")
    benchmark_results = engine.benchmark_optimization()

    # Print summary
    summary = engine.get_performance_summary()
    logger.info("📊 Performance Summary:")
    logger.info(".2f")
    logger.info(".2f")
    logger.info(".2f")
    logger.info(".1f")

    logger.info("✅ OpenCL optimization test completed")


if __name__ == "__main__":
    main()
