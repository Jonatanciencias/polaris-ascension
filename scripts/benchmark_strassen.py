#!/usr/bin/env python3
"""
Strassen Matrix Multiplication Benchmark
Phase 2, Technique 4: Advanced Algorithm Research

Evaluates theoretical O(n^2.807) vs practical GPU performance
Compares Strassen variants against Phase 1 baseline implementations

Hardware: AMD Radeon RX 590 (Polaris 10)
Expected: Assessment of Strassen's practical viability on GPU
"""

import os
import sys
import time
import numpy as np
import pyopencl as cl
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class StrassenBenchmark:
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.kernels = {}
        self.results = {}

    def initialize_opencl(self):
        """Initialize OpenCL context and command queue"""
        try:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            print("✓ OpenCL initialized successfully")
            return True
        except Exception as e:
            print(f"✗ OpenCL initialization failed: {e}")
            return False

    def load_kernel(self, kernel_file: str, kernel_name: str) -> bool:
        """Load and compile a kernel from file"""
        try:
            kernel_path = os.path.join('src', 'opencl', 'kernels', kernel_file)
            with open(kernel_path, 'r') as f:
                source = f.read()

            program = cl.Program(self.ctx, source).build()

            if hasattr(program, kernel_name):
                self.kernels[kernel_name] = getattr(program, kernel_name)
                print(f"✓ Kernel {kernel_name} loaded successfully")
                return True
            else:
                print(f"✗ Kernel {kernel_name} not found in {kernel_file}")
                return False

        except Exception as e:
            print(f"✗ Failed to load kernel {kernel_name}: {e}")
            return False

    def load_all_kernels(self) -> bool:
        """Load all kernels for benchmarking"""
        kernels_to_load = [
            ('gemm_strassen.cl', 'gemm_strassen_complete'),
            ('gemm_strassen.cl', 'gemm_strassen_simple'),
            ('gemm.cl', 'gemm_tiled'),  # Phase 1 baseline
            ('gemm.cl', 'gemm_vectorized_float4'),  # Phase 1 optimized
        ]

        success = True
        for kernel_file, kernel_name in kernels_to_load:
            if not self.load_kernel(kernel_file, kernel_name):
                success = False

        return success

    def benchmark_kernel(self, kernel_name: str, M: int, N: int, K: int,
                        num_runs: int = 5) -> Dict[str, float]:
        """Benchmark a single kernel execution"""
        if kernel_name not in self.kernels:
            return {'error': f'Kernel {kernel_name} not loaded'}

        kernel = self.kernels[kernel_name]

        # Create test data (ensure even dimensions for 2x2 blocking)
        M_adj = M if M % 2 == 0 else M + 1
        N_adj = N if N % 2 == 0 else N + 1

        A = np.random.randn(M_adj, K).astype(np.float32)
        B = np.random.randn(K, N_adj).astype(np.float32)
        C = np.zeros((M_adj, N_adj), dtype=np.float32)

        # OpenCL buffers
        A_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

        # Set kernel arguments
        kernel.set_args(np.int32(M_adj), np.int32(N_adj), np.int32(K),
                       np.float32(1.0), np.float32(0.0), A_buf, B_buf, C_buf)

        # Determine work group size based on kernel type
        if 'strassen' in kernel_name:
            # 2x2 blocking for Strassen
            global_size = (M_adj//2, N_adj//2)
            local_size = (1, 1)
        else:
            # Standard tiled kernel - number of work groups
            tile_size = 32
            num_groups_x = (M_adj + tile_size - 1) // tile_size
            num_groups_y = (N_adj + tile_size - 1) // tile_size
            global_size = (num_groups_x, num_groups_y)
            local_size = (1, 1)  # Simplified for testing

        # Warmup run
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()

        # Benchmark runs
        times = []
        for _ in range(num_runs):
            self.queue.finish()
            start = time.time()

            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()

            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds

        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)

        # Calculate performance metrics
        operations = 2 * M_adj * N_adj * K  # FLOPs for GEMM
        gflops = (operations / (avg_time / 1000)) / 1e9

        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'matrix_size': f'{M_adj}x{N_adj}x{K}',
            'kernel': kernel_name
        }

    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark comparing all implementations"""
        print("\n=== Strassen Matrix Multiplication Benchmark ===")
        print("Phase 2, Technique 4: Advanced Algorithm Research")
        print("Hardware: AMD Radeon RX 590 (Polaris 10)")

        # Test matrix sizes (powers of 2 for Strassen compatibility)
        sizes = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ]

        kernels_to_test = [
            'gemm_tiled',  # Phase 1 baseline
            'gemm_vectorized_float4',  # Phase 1 optimized
            'gemm_strassen_simple',  # Strassen with classical fallback
            'gemm_strassen_complete',  # Full Strassen implementation
        ]

        results = {}

        for M, N, K in sizes:
            print(f"\n--- Testing {M}x{N}x{K} matrices ---")
            size_key = f'{M}x{N}x{K}'
            results[size_key] = {}

            for kernel_name in kernels_to_test:
                if kernel_name in self.kernels:
                    print(f"  Benchmarking {kernel_name}...")
                    result = self.benchmark_kernel(kernel_name, M, N, K)
                    results[size_key][kernel_name] = result

                    if 'error' not in result:
                        print(f"    {kernel_name}: {result['gflops']:.2f} GFLOPS")
                else:
                    print(f"  Skipping {kernel_name} (not loaded)")
                    results[size_key][kernel_name] = {'error': 'Kernel not loaded'}

        return results

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze benchmark results and provide insights"""
        analysis = {
            'strassen_vs_classical': {},
            'performance_summary': {},
            'recommendations': []
        }

        for size_key, size_results in results.items():
            analysis['strassen_vs_classical'][size_key] = {}

            # Compare Strassen variants against Phase 1 baseline
            baseline_gflops = None
            strassen_gflops = {}

            for kernel_name, result in size_results.items():
                if 'error' in result:
                    continue

                gflops = result.get('gflops', 0)

                if kernel_name == 'gemm_tiled':
                    baseline_gflops = gflops
                elif 'strassen' in kernel_name:
                    strassen_gflops[kernel_name] = gflops

            if baseline_gflops:
                for strassen_kernel, strassen_gflops_val in strassen_gflops.items():
                    speedup = strassen_gflops_val / baseline_gflops
                    analysis['strassen_vs_classical'][size_key][strassen_kernel] = {
                        'speedup': speedup,
                        'baseline_gflops': baseline_gflops,
                        'strassen_gflops': strassen_gflops_val
                    }

        # Overall assessment
        total_strassen_speedup = []
        for size_key, comparisons in analysis['strassen_vs_classical'].items():
            for kernel, comp in comparisons.items():
                total_strassen_speedup.append(comp['speedup'])

        if total_strassen_speedup:
            avg_speedup = np.mean(total_strassen_speedup)
            analysis['performance_summary'] = {
                'average_strassen_speedup': avg_speedup,
                'strassen_theoretical_limit': 2.807,  # log2(7)
                'practical_viability': 'Viable' if avg_speedup > 0.8 else 'Not recommended'
            }

            if avg_speedup < 1.0:
                analysis['recommendations'].append(
                    "Strassen implementation shows performance regression. "
                    "Theoretical O(n^2.807) benefits not realized on Polaris 10 architecture."
                )
            else:
                analysis['recommendations'].append(
                    "Strassen shows performance benefits. Consider for larger matrices."
                )

        return analysis

    def save_results(self, results: Dict, analysis: Dict, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strassen_benchmark_{timestamp}.json"

        output = {
            'benchmark_info': {
                'phase': 'Phase 2, Technique 4',
                'technique': 'Advanced Algorithm Research',
                'hardware': 'AMD Radeon RX 590 (Polaris 10)',
                'timestamp': datetime.now().isoformat(),
                'objective': 'Evaluate Strassen O(n^2.807) vs practical GPU performance'
            },
            'results': results,
            'analysis': analysis
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✓ Results saved to {filename}")

    def print_summary(self, results: Dict, analysis: Dict):
        """Print human-readable benchmark summary"""
        print("\n" + "="*60)
        print("STRASSEN MATRIX MULTIPLICATION BENCHMARK SUMMARY")
        print("="*60)

        print("\n🎯 OBJECTIVE:")
        print("   Evaluate theoretical O(n^2.807) complexity benefits")
        print("   vs practical performance on AMD Radeon RX 590")

        print("\n📊 PERFORMANCE SUMMARY:")
        if 'performance_summary' in analysis:
            summary = analysis['performance_summary']
            print(f"   Average Strassen Speedup: {summary['average_strassen_speedup']:.3f}x")
            print(f"   Theoretical Limit: O(n^{summary['strassen_theoretical_limit']})")
            print(f"   Practical Viability: {summary['practical_viability']}")

        print("\n🔍 DETAILED COMPARISONS:")
        for size_key, comparisons in analysis['strassen_vs_classical'].items():
            print(f"\n   Matrix Size: {size_key}")
            for kernel, comp in comparisons.items():
                speedup = comp['speedup']
                status = "✅ BETTER" if speedup > 1.0 else "❌ WORSE"
                print(f"    {kernel}: {speedup:.3f}x speedup {status}")
        print("\n💡 RECOMMENDATIONS:")
        for rec in analysis.get('recommendations', []):
            print(f"   • {rec}")

        print("\n" + "="*60)

def main():
    """Main benchmark execution"""
    benchmark = StrassenBenchmark()

    # Initialize OpenCL
    if not benchmark.initialize_opencl():
        sys.exit(1)

    # Load kernels
    if not benchmark.load_all_kernels():
        print("Warning: Some kernels failed to load, continuing with available ones...")

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Analyze results
    analysis = benchmark.analyze_results(results)

    # Save and display results
    benchmark.save_results(results, analysis)
    benchmark.print_summary(results, analysis)

    print("\n✓ Strassen evaluation complete!")
    print("  Results demonstrate the practical limits of Strassen's algorithm")
    print("  on modern GPU architectures with limited memory bandwidth.")

if __name__ == '__main__':
    main()