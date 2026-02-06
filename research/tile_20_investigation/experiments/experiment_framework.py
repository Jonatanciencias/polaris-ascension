#!/usr/bin/env python3
"""
Experiment Framework for Tile=20 Investigation

Provides common testing infrastructure for all experimental approaches.
Ensures consistent benchmarking and correctness validation.
"""

import numpy as np
import pyopencl as cl
import time
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    approach_name: str
    matrix_size: int
    performance_gflops: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    max_error: float
    is_correct: bool
    has_nan: bool
    has_inf: bool
    status: str  # "SUCCESS", "CORRECTNESS_FAIL", "PERFORMANCE_FAIL", "ERROR"
    error_message: Optional[str] = None


class ExperimentFramework:
    """Framework for running tile=20 experiments"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.setup_opencl()
        
    def setup_opencl(self):
        """Setup OpenCL context and queue"""
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
        self.device = devices[0]
        self.ctx = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(
            self.ctx, 
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        
        if self.verbose:
            print("=" * 70)
            print("🔬 TILE=20 EXPERIMENT FRAMEWORK")
            print("=" * 70)
            print(f"Device: {self.device.name}")
            print(f"OpenCL Version: {self.device.version}")
            print(f"Max Work Group Size: {self.device.max_work_group_size}")
            print(f"Local Memory: {self.device.local_mem_size / 1024:.0f} KB")
            print("=" * 70)
            print()
    
    def load_kernel(self, kernel_file: str, kernel_name: str):
        """Load and compile a kernel from file"""
        if not os.path.exists(kernel_file):
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")
        
        with open(kernel_file, 'r') as f:
            kernel_source = f.read()
        
        try:
            program = cl.Program(self.ctx, kernel_source).build()
            kernel = getattr(program, kernel_name)
            return kernel
        except Exception as e:
            print(f"❌ Kernel compilation failed: {e}")
            raise
    
    def create_test_matrices(self, M: int, N: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create test matrices with known values"""
        # Use small random values for numerical stability
        A = np.random.randn(M, K).astype(np.float32) * 0.1
        B = np.random.randn(K, N).astype(np.float32) * 0.1
        C = np.zeros((M, N), dtype=np.float32)
        return A, B, C
    
    def compute_reference(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute reference result using NumPy"""
        return (A @ B).astype(np.float32)
    
    def check_correctness(self, C: np.ndarray, C_ref: np.ndarray, 
                         tolerance: float = 0.1) -> Tuple[bool, float, bool, bool]:
        """
        Check correctness of result
        
        Returns:
            (is_correct, max_error, has_nan, has_inf)
        """
        has_nan = np.any(np.isnan(C))
        has_inf = np.any(np.isinf(C))
        
        if has_nan or has_inf:
            return False, float('inf'), has_nan, has_inf
        
        error = np.abs(C - C_ref)
        max_error = np.max(error)
        is_correct = max_error < tolerance
        
        return is_correct, max_error, has_nan, has_inf
    
    def benchmark_kernel(self, kernel, global_size: Tuple[int, int], 
                        local_size: Tuple[int, int], args: list,
                        warmup_iters: int = 3, bench_iters: int = 10) -> Tuple[float, float, float]:
        """
        Benchmark kernel execution
        
        Returns:
            (avg_time_ms, min_time_ms, max_time_ms)
        """
        # Warmup
        for _ in range(warmup_iters):
            kernel(self.queue, global_size, local_size, *args)
        self.queue.finish()
        
        # Benchmark
        times = []
        for _ in range(bench_iters):
            event = kernel(self.queue, global_size, local_size, *args)
            event.wait()
            time_ns = event.profile.end - event.profile.start
            times.append(time_ns * 1e-6)  # Convert to milliseconds
        
        return np.mean(times), np.min(times), np.max(times)
    
    def run_experiment(self, 
                      approach_name: str,
                      kernel_file: str,
                      kernel_name: str,
                      matrix_size: int,
                      get_work_size_func,
                      get_kernel_args_func,
                      correctness_tolerance: float = 0.1) -> ExperimentResult:
        """
        Run a complete experiment
        
        Args:
            approach_name: Name of the approach being tested
            kernel_file: Path to kernel .cl file
            kernel_name: Name of kernel function
            matrix_size: Size of test matrix (M=N=K)
            get_work_size_func: Function(M,N,K) -> (global_size, local_size)
            get_kernel_args_func: Function(M,N,K,A_buf,B_buf,C_buf) -> kernel_args_list
            correctness_tolerance: Maximum allowed error
        
        Returns:
            ExperimentResult with all metrics
        """
        M = N = K = matrix_size
        
        try:
            # Load kernel
            if self.verbose:
                print(f"📦 Loading kernel: {kernel_name} from {os.path.basename(kernel_file)}")
            kernel = self.load_kernel(kernel_file, kernel_name)
            
            # Create test data
            if self.verbose:
                print(f"🔢 Creating test matrices ({M}×{K}, {K}×{N})")
            A, B, C = self.create_test_matrices(M, N, K)
            C_ref = self.compute_reference(A, B)
            
            # Create buffers
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)
            
            # Get work size
            global_size, local_size = get_work_size_func(M, N, K)
            if self.verbose:
                print(f"⚙️  Work size: global={global_size}, local={local_size}")
            
            # Get kernel arguments
            kernel_args = get_kernel_args_func(M, N, K, A_buf, B_buf, C_buf)
            
            # Benchmark
            if self.verbose:
                print(f"⏱️  Benchmarking...")
            avg_time, min_time, max_time = self.benchmark_kernel(
                kernel, global_size, local_size, kernel_args
            )
            
            # Read result
            cl.enqueue_copy(self.queue, C, C_buf).wait()
            
            # Check correctness
            is_correct, max_error, has_nan, has_inf = self.check_correctness(
                C, C_ref, correctness_tolerance
            )
            
            # Calculate performance
            flops = 2 * M * N * K
            gflops = (flops / (avg_time * 1e-3)) / 1e9
            
            # Determine status
            if has_nan or has_inf:
                status = "ERROR"
                error_msg = f"NaN={has_nan}, Inf={has_inf}"
            elif not is_correct:
                status = "CORRECTNESS_FAIL"
                error_msg = f"max_error={max_error:.4f} > tolerance={correctness_tolerance}"
            else:
                status = "SUCCESS"
                error_msg = None
            
            result = ExperimentResult(
                approach_name=approach_name,
                matrix_size=matrix_size,
                performance_gflops=gflops,
                avg_time_ms=avg_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                max_error=max_error,
                is_correct=is_correct,
                has_nan=has_nan,
                has_inf=has_inf,
                status=status,
                error_message=error_msg
            )
            
            # Print result
            if self.verbose:
                self.print_result(result)
            
            return result
            
        except Exception as e:
            return ExperimentResult(
                approach_name=approach_name,
                matrix_size=matrix_size,
                performance_gflops=0.0,
                avg_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                max_error=float('inf'),
                is_correct=False,
                has_nan=False,
                has_inf=False,
                status="ERROR",
                error_message=str(e)
            )
    
    def print_result(self, result: ExperimentResult):
        """Print experiment result"""
        print()
        print("=" * 70)
        print(f"📊 EXPERIMENT RESULT: {result.approach_name}")
        print("=" * 70)
        print(f"Matrix Size:     {result.matrix_size}×{result.matrix_size}")
        print(f"Performance:     {result.performance_gflops:.2f} GFLOPS")
        print(f"Avg Time:        {result.avg_time_ms:.2f} ms")
        print(f"Min Time:        {result.min_time_ms:.2f} ms")
        print(f"Max Time:        {result.max_time_ms:.2f} ms")
        print(f"Max Error:       {result.max_error:.6f}")
        print(f"Correctness:     {'✅ PASS' if result.is_correct else '❌ FAIL'}")
        print(f"Has NaN:         {'❌ YES' if result.has_nan else '✅ NO'}")
        print(f"Has Inf:         {'❌ YES' if result.has_inf else '✅ NO'}")
        print(f"Status:          {result.status}")
        if result.error_message:
            print(f"Error:           {result.error_message}")
        print("=" * 70)
        print()
    
    def run_size_sweep(self, 
                      approach_name: str,
                      kernel_file: str,
                      kernel_name: str,
                      sizes: list,
                      get_work_size_func,
                      get_kernel_args_func) -> list:
        """Run experiment across multiple matrix sizes"""
        results = []
        
        print()
        print("=" * 70)
        print(f"🔬 SIZE SWEEP: {approach_name}")
        print("=" * 70)
        print()
        
        for size in sizes:
            result = self.run_experiment(
                approach_name=f"{approach_name}_{size}",
                kernel_file=kernel_file,
                kernel_name=kernel_name,
                matrix_size=size,
                get_work_size_func=get_work_size_func,
                get_kernel_args_func=get_kernel_args_func
            )
            results.append(result)
        
        # Print summary
        print()
        print("=" * 70)
        print(f"📈 SUMMARY: {approach_name}")
        print("=" * 70)
        print(f"{'Size':>8} | {'GFLOPS':>10} | {'Time (ms)':>10} | {'Error':>12} | {'Status':>10}")
        print("-" * 70)
        for r in results:
            status_icon = "✅" if r.status == "SUCCESS" else "❌"
            print(f"{r.matrix_size:>8} | {r.performance_gflops:>10.2f} | "
                  f"{r.avg_time_ms:>10.2f} | {r.max_error:>12.6f} | "
                  f"{status_icon} {r.status}")
        print("=" * 70)
        print()
        
        return results


def compare_approaches(results_dict: dict):
    """
    Compare multiple approaches
    
    Args:
        results_dict: {approach_name: [ExperimentResult, ...]}
    """
    print()
    print("=" * 80)
    print("🏆 APPROACH COMPARISON")
    print("=" * 80)
    print()
    
    # Get all unique sizes
    all_sizes = set()
    for results in results_dict.values():
        for r in results:
            all_sizes.add(r.matrix_size)
    all_sizes = sorted(all_sizes)
    
    # Compare by size
    for size in all_sizes:
        print(f"\n📊 Matrix Size: {size}×{size}")
        print("-" * 80)
        print(f"{'Approach':>25} | {'GFLOPS':>10} | {'Error':>12} | {'Status':>10}")
        print("-" * 80)
        
        size_results = []
        for approach_name, results in results_dict.items():
            matching = [r for r in results if r.matrix_size == size]
            if matching:
                size_results.append((approach_name, matching[0]))
        
        # Sort by performance
        size_results.sort(key=lambda x: x[1].performance_gflops, reverse=True)
        
        for approach_name, result in size_results:
            status_icon = "✅" if result.status == "SUCCESS" else "❌"
            print(f"{approach_name:>25} | {result.performance_gflops:>10.2f} | "
                  f"{result.max_error:>12.6f} | {status_icon} {result.status}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    print("🔬 Experiment Framework Ready")
    print()
    print("Usage:")
    print("  from experiment_framework import ExperimentFramework")
    print("  framework = ExperimentFramework()")
    print("  result = framework.run_experiment(...)")
