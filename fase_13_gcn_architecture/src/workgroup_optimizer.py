#!/usr/bin/env python3
"""
🚀 WORK-GROUP OPTIMIZER FOR RADEON RX 580
=========================================

Optimizador automático de tamaños de work-group para arquitectura GCN 4.0.
Implementa la configuración óptima (4, 64) identificada por el analizador.

Características:
- Auto-tuning de work-group sizes
- Validation de configuraciones óptimas
- Performance benchmarking
- Occupancy analysis

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pyopencl as cl
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WorkGroupConfig:
    """Configuración de work-group con métricas"""
    size: Tuple[int, int]
    gflops: float
    kernel_time_ms: float
    occupancy: float
    efficiency: float
    valid: bool

@dataclass
class OptimizationResult:
    """Resultado de optimización de work-groups"""
    optimal_config: WorkGroupConfig
    tested_configs: List[WorkGroupConfig]
    improvement_factor: float
    baseline_gflops: float
    optimal_gflops: float
    validation_passed: bool

class WorkGroupOptimizer:
    """
    Work-Group Optimizer for Radeon RX 580

    Automatically finds optimal work-group sizes for GCN 4.0 architecture
    through systematic testing and performance analysis.
    """

    def __init__(self, context: cl.Context, device: cl.Device, queue: cl.CommandQueue):
        self.context = context
        self.device = device
        self.queue = queue
        self.results_dir = "results"

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # GCN 4.0 specific configurations to test
        self.test_configs = [
            (4, 64),   # Optimal from analysis
            (8, 32),   # Rectangular options
            (16, 16),  # Square (current default)
            (32, 8),   # Wide
            (64, 4),   # Very wide
            (8, 8),    # Small square
            (2, 128),  # Extreme rectangular
            (128, 2),  # Other extreme
        ]

        logger.info("🚀 Work-Group Optimizer initialized")

    def optimize_work_groups(self, kernel_source: str, test_matrix_size: int = 1024) -> OptimizationResult:
        """
        Find optimal work-group configuration through systematic testing

        Args:
            kernel_source: OpenCL kernel source code
            test_matrix_size: Size of test matrices (NxN)

        Returns:
            OptimizationResult with optimal configuration
        """
        logger.info(f"🔍 Optimizing work-groups for {test_matrix_size}x{test_matrix_size} matrices...")

        # Build kernel program
        program = cl.Program(self.context, kernel_source).build()

        # Test configurations
        results = []
        baseline_gflops = None

        for wg_size in self.test_configs:
            try:
                config_result = self._test_work_group_config(
                    program, wg_size, test_matrix_size
                )

                if config_result.valid:
                    results.append(config_result)

                    # Use (16,16) as baseline
                    if wg_size == (16, 16):
                        baseline_gflops = config_result.gflops

                    logger.info(f"Work-group {wg_size}: {config_result.gflops:.2f} GFLOPS "
                              f"(occupancy: {config_result.occupancy:.1%})")

            except Exception as e:
                logger.warning(f"Failed to test work-group {wg_size}: {e}")
                continue

        if not results:
            raise RuntimeError("No valid work-group configurations found")

        # Find optimal configuration
        optimal_config = max(results, key=lambda x: x.gflops)

        # Calculate improvement
        if baseline_gflops is None:
            baseline_gflops = results[0].gflops  # Use first valid result as baseline

        improvement_factor = optimal_config.gflops / baseline_gflops

        # Validate optimal configuration
        validation_passed = self._validate_optimal_config(optimal_config, program, test_matrix_size)

        result = OptimizationResult(
            optimal_config=optimal_config,
            tested_configs=results,
            improvement_factor=improvement_factor,
            baseline_gflops=baseline_gflops,
            optimal_gflops=optimal_config.gflops,
            validation_passed=validation_passed
        )

        logger.info(f"✅ Optimal work-group: {optimal_config.size} "
                   f"({optimal_config.gflops:.2f} GFLOPS, {improvement_factor:.2f}x improvement)")

        return result

    def _test_work_group_config(self, program: cl.Program,
                               wg_size: Tuple[int, int],
                               matrix_size: int) -> WorkGroupConfig:
        """
        Test a specific work-group configuration

        Args:
            program: Compiled OpenCL program
            wg_size: Work-group size to test
            matrix_size: Size of test matrices

        Returns:
            WorkGroupConfig with performance metrics
        """
        M = N = K = matrix_size

        # Generate test data
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=C.nbytes)

        # Get kernel
        kernel = program.gemm_fp32
        kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

        # Check if work-group size is valid
        wg_elements = wg_size[0] * wg_size[1]
        if wg_elements > self.device.max_work_group_size:
            return WorkGroupConfig(
                size=wg_size,
                gflops=0.0,
                kernel_time_ms=0.0,
                occupancy=0.0,
                efficiency=0.0,
                valid=False
            )

        global_size = (M, N)
        local_size = wg_size

        # Warm up
        try:
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()
        except Exception:
            return WorkGroupConfig(
                size=wg_size,
                gflops=0.0,
                kernel_time_ms=0.0,
                occupancy=0.0,
                efficiency=0.0,
                valid=False
            )

        # Benchmark (multiple runs for stability)
        times = []
        for _ in range(5):
            start_time = time.time()
            event = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            event.wait()
            end_time = time.time()

            kernel_time = (event.profile.end - event.profile.start) * 1e-9
            times.append(kernel_time)

        avg_kernel_time = np.mean(times)

        # Calculate performance
        operations = 2 * M * N * K
        gflops = operations / (avg_kernel_time * 1e9)

        # Calculate occupancy (simplified GCN model)
        wavefront_size = 64  # GCN wavefront size
        work_items_per_wavefront = wavefront_size // wg_elements
        if work_items_per_wavefront < 1:
            work_items_per_wavefront = 1

        # Rough occupancy estimate
        occupancy = min(1.0, (wg_elements / wavefront_size))

        # Efficiency relative to theoretical peak
        theoretical_peak = 36 * 1050 * 2 * 64 / 1000  # Rough estimate for Polaris 10
        efficiency = gflops / theoretical_peak

        return WorkGroupConfig(
            size=wg_size,
            gflops=gflops,
            kernel_time_ms=avg_kernel_time * 1000,
            occupancy=occupancy,
            efficiency=efficiency,
            valid=True
        )

    def _validate_optimal_config(self, config: WorkGroupConfig,
                                program: cl.Program, matrix_size: int) -> bool:
        """
        Validate that optimal configuration produces correct results

        Args:
            config: Work-group configuration to validate
            program: OpenCL program
            matrix_size: Test matrix size

        Returns:
            True if validation passes
        """
        try:
            M = N = K = matrix_size

            # Generate test data
            np.random.seed(42)  # For reproducible results
            A = np.random.rand(M, K).astype(np.float32)
            B = np.random.rand(K, N).astype(np.float32)

            # Reference result (NumPy)
            C_ref = np.dot(A, B)

            # GPU result with optimal config
            mf = cl.mem_flags
            A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, size=C_ref.nbytes)

            kernel = program.gemm_fp32
            kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

            global_size = (M, N)
            local_size = config.size

            event = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            event.wait()

            C_gpu = np.empty_like(C_ref)
            cl.enqueue_copy(self.queue, C_gpu, C_buf)

            # Check accuracy
            max_error = np.max(np.abs(C_gpu - C_ref))
            relative_error = max_error / np.max(np.abs(C_ref))

            # Validation passes if relative error < 1e-5
            validation_passed = relative_error < 1e-5

            if validation_passed:
                logger.info(f"✅ Validation passed for work-group {config.size} "
                          f"(relative error: {relative_error:.2e})")
            else:
                logger.warning(f"❌ Validation failed for work-group {config.size} "
                             f"(relative error: {relative_error:.2e})")

            return validation_passed

        except Exception as e:
            logger.error(f"Validation failed for work-group {config.size}: {e}")
            return False

    def generate_optimized_kernel(self, base_kernel_source: str,
                                optimal_config: WorkGroupConfig) -> str:
        """
        Generate kernel optimized for the optimal work-group configuration

        Args:
            base_kernel_source: Base kernel source
            optimal_config: Optimal work-group configuration

        Returns:
            Optimized kernel source
        """
        # For now, return the base kernel (work-group size is set at launch)
        # Future optimizations could include work-group specific optimizations
        return base_kernel_source

    def save_results(self, result: OptimizationResult, filename: str = "workgroup_optimization_results.json"):
        """
        Save optimization results to file

        Args:
            result: Optimization results
            filename: Output filename
        """
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimal_config': {
                'size': list(result.optimal_config.size),  # Convert tuple to list
                'gflops': result.optimal_config.gflops,
                'kernel_time_ms': result.optimal_config.kernel_time_ms,
                'occupancy': result.optimal_config.occupancy,
                'efficiency': result.optimal_config.efficiency
            },
            'tested_configs': [
                {
                    'size': list(config.size),  # Convert tuple to list
                    'gflops': config.gflops,
                    'kernel_time_ms': config.kernel_time_ms,
                    'occupancy': config.occupancy,
                    'efficiency': config.efficiency,
                    'valid': bool(config.valid)  # Ensure bool type
                }
                for config in result.tested_configs
            ],
            'metrics': {
                'improvement_factor': result.improvement_factor,
                'baseline_gflops': result.baseline_gflops,
                'optimal_gflops': result.optimal_gflops,
                'validation_passed': bool(result.validation_passed)  # Ensure bool type
            }
        }

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ Results saved to {filepath}")

    def print_summary(self, result: OptimizationResult):
        """Print human-readable summary of optimization results"""
        print("\n" + "="*80)
        print("🚀 WORK-GROUP OPTIMIZATION RESULTS")
        print("="*80)

        opt = result.optimal_config
        print(f"\n🎯 OPTIMAL CONFIGURATION:")
        print(f"   Work-Group Size: {opt.size}")
        print(f"   Performance: {opt.gflops:.2f} GFLOPS")
        print(f"   Kernel Time: {opt.kernel_time_ms:.2f} ms")
        print(f"   Occupancy: {opt.occupancy:.1%}")
        print(f"   Efficiency: {opt.efficiency:.1%}")

        print(f"\n📊 IMPROVEMENT METRICS:")
        print(f"   Baseline: {result.baseline_gflops:.2f} GFLOPS")
        print(f"   Optimal: {result.optimal_gflops:.2f} GFLOPS")
        print(f"   Improvement: {result.improvement_factor:.2f}x ({(result.improvement_factor-1)*100:.1f}%)")

        print(f"\n🔧 TESTED CONFIGURATIONS:")
        for config in sorted(result.tested_configs, key=lambda x: x.gflops, reverse=True):
            status = "✅" if config.valid else "❌"
            print(f"   {status} {config.size}: {config.gflops:.2f} GFLOPS")

        validation_status = "✅ PASSED" if result.validation_passed else "❌ FAILED"
        print(f"\n🧪 VALIDATION: {validation_status}")

        print("\n" + "="*80)

def main():
    """Main function for work-group optimization"""
    try:
        # Initialize OpenCL
        platforms = cl.get_platforms()
        amd_platforms = [p for p in platforms if 'AMD' in p.name or 'Advanced Micro Devices' in p.name]
        platform = amd_platforms[0] if amd_platforms else platforms[0]

        devices = platform.get_devices(device_type=cl.device_type.GPU)
        radeon_devices = [d for d in devices if 'Radeon RX 580' in d.name or '580' in d.name]
        device = radeon_devices[0] if radeon_devices else devices[0]

        context = cl.Context([device])
        queue = cl.CommandQueue(context, device, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Create optimizer
        optimizer = WorkGroupOptimizer(context, device, queue)

        # Base GEMM kernel
        kernel_source = """
        __kernel void gemm_fp32(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int M, const int N, const int K)
        {
            const int row = get_global_id(0);
            const int col = get_global_id(1);

            if (row >= M || col >= N) return;

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        """

        # Run optimization
        result = optimizer.optimize_work_groups(kernel_source, test_matrix_size=1024)

        # Save and display results
        optimizer.save_results(result)
        optimizer.print_summary(result)

        logger.info("🎯 Work-group optimization completed successfully!")

    except Exception as e:
        logger.error(f"Work-group optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()