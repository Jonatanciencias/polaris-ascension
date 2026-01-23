#!/usr/bin/env python3
"""
GPU Stress Test with Power Monitoring
======================================

Generate synthetic GPU workload at different intensities and monitor
real-time power consumption using kernel sensors.

This demonstrates the power profiling framework's ability to track
GPU power consumption across different workload intensities.

Usage:
    python scripts/gpu_stress_test.py --duration 60 --output results/gpu_power_profile.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import threading
from datetime import datetime
from typing import List, Dict
import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("❌ PyOpenCL not available. Install with: pip install pyopencl")
    sys.exit(1)

from scripts.power_monitor import GPUPowerMonitor


class GPUStressTest:
    """Generate GPU workload and monitor power consumption."""
    
    # OpenCL kernel for matrix multiplication (compute-intensive)
    KERNEL_SOURCE = """
    __kernel void matrix_multiply(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int N)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);
        
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    
    __kernel void vector_add(
        __global const float* A,
        __global const float* B,
        __global float* C)
    {
        int idx = get_global_id(0);
        C[idx] = A[idx] + B[idx];
    }
    
    __kernel void mandelbrot(
        __global float* output,
        const int width,
        const int height,
        const int max_iter)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        
        if (x >= width || y >= height) return;
        
        float cr = (x - width/2.0f) * 4.0f / width;
        float ci = (y - height/2.0f) * 4.0f / height;
        
        float zr = 0.0f, zi = 0.0f;
        int iter = 0;
        
        while (iter < max_iter && (zr*zr + zi*zi) < 4.0f) {
            float temp = zr*zr - zi*zi + cr;
            zi = 2.0f*zr*zi + ci;
            zr = temp;
            iter++;
        }
        
        output[y * width + x] = (float)iter / max_iter;
    }
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize GPU stress test."""
        self.verbose = verbose
        self.power_monitor = GPUPowerMonitor(verbose=verbose)
        
        # Initialize OpenCL
        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            # Find AMD GPU
            self.device = None
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    if 'AMD' in device.vendor or 'Radeon' in device.name:
                        self.device = device
                        self.platform = platform
                        break
                if self.device:
                    break
            
            if not self.device:
                raise RuntimeError("No AMD GPU found")
            
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            self.program = cl.Program(self.context, self.KERNEL_SOURCE).build()
            
            if self.verbose:
                print(f"✅ OpenCL initialized")
                print(f"   Platform: {self.platform.name}")
                print(f"   Device: {self.device.name}")
                print(f"   Max compute units: {self.device.max_compute_units}")
                print(f"   Max work group size: {self.device.max_work_group_size}")
            
        except Exception as e:
            print(f"❌ Failed to initialize OpenCL: {e}")
            raise
    
    def matrix_multiply_workload(self, size: int = 1024, iterations: int = 100):
        """Run matrix multiplication workload."""
        # Create random matrices
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, C.nbytes)
        
        # Execute kernel multiple times
        for _ in range(iterations):
            self.program.matrix_multiply(
                self.queue, (size, size), None,
                A_buf, B_buf, C_buf, np.int32(size)
            )
        
        self.queue.finish()
    
    def mandelbrot_workload(self, width: int = 2048, height: int = 2048, 
                           max_iter: int = 1000, iterations: int = 50):
        """Run Mandelbrot set computation (complex arithmetic)."""
        output = np.zeros((height, width), dtype=np.float32)
        
        mf = cl.mem_flags
        output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
        
        for _ in range(iterations):
            self.program.mandelbrot(
                self.queue, (width, height), None,
                output_buf, np.int32(width), np.int32(height), np.int32(max_iter)
            )
        
        self.queue.finish()
    
    def vector_add_workload(self, size: int = 10_000_000, iterations: int = 1000):
        """Run vector addition (memory-intensive)."""
        A = np.random.randn(size).astype(np.float32)
        B = np.random.randn(size).astype(np.float32)
        C = np.zeros(size, dtype=np.float32)
        
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, C.nbytes)
        
        for _ in range(iterations):
            self.program.vector_add(self.queue, (size,), None, A_buf, B_buf, C_buf)
        
        self.queue.finish()
    
    def run_workload_with_monitoring(self, workload_fn, workload_name: str, 
                                     duration: int = 30) -> Dict:
        """
        Run workload while monitoring power.
        
        Args:
            workload_fn: Function to generate GPU workload
            workload_name: Name of the workload
            duration: Duration in seconds
            
        Returns:
            Dictionary with power statistics
        """
        print(f"\n{'='*70}")
        print(f"Workload: {workload_name}")
        print(f"Duration: {duration}s")
        print(f"{'='*70}")
        
        power_readings = []
        stop_flag = threading.Event()
        
        def monitor_power():
            """Background thread for power monitoring."""
            while not stop_flag.is_set():
                reading = self.power_monitor.read_full()
                power_readings.append(reading)
                if self.verbose:
                    print(f"  Power: {reading.power_watts:.2f}W  Temp: {reading.temperature}°C")
                time.sleep(0.1)  # 10Hz sampling
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_power, daemon=True)
        monitor_thread.start()
        
        # Run workload
        start_time = time.time()
        iterations = 0
        
        try:
            while time.time() - start_time < duration:
                workload_fn()
                iterations += 1
        finally:
            stop_flag.set()
            monitor_thread.join(timeout=2)
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        stats = self.power_monitor.calculate_statistics(power_readings)
        
        result = {
            'workload': workload_name,
            'duration_seconds': elapsed,
            'iterations': iterations,
            'mean_power_watts': stats.mean_power,
            'min_power_watts': stats.min_power,
            'max_power_watts': stats.max_power,
            'std_power_watts': stats.std_power,
            'total_energy_joules': stats.total_energy_joules,
            'avg_temperature': stats.avg_temperature,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n📊 Results:")
        print(f"   Iterations:    {iterations}")
        print(f"   Mean Power:    {stats.mean_power:.2f} W")
        print(f"   Min Power:     {stats.min_power:.2f} W")
        print(f"   Max Power:     {stats.max_power:.2f} W")
        print(f"   Std Dev:       {stats.std_power:.2f} W")
        print(f"   Total Energy:  {stats.total_energy_joules:.2f} J")
        if stats.avg_temperature:
            print(f"   Avg Temp:      {stats.avg_temperature:.1f} °C")
        
        return result
    
    def run_intensity_sweep(self, duration_per_level: int = 30) -> List[Dict]:
        """
        Run workloads at different intensity levels.
        
        Args:
            duration_per_level: Duration for each intensity level
            
        Returns:
            List of results for each intensity level
        """
        results = []
        
        print(f"\n{'='*70}")
        print(f"GPU Power Profiling - Intensity Sweep")
        print(f"{'='*70}")
        print(f"Duration per level: {duration_per_level}s")
        print(f"GPU: {self.device.name}")
        print(f"{'='*70}")
        
        # Level 1: Idle (no workload)
        print(f"\n[1/5] Idle (no GPU load)")
        time.sleep(2)
        result = self.run_workload_with_monitoring(
            lambda: time.sleep(0.1),
            "Idle",
            duration_per_level
        )
        results.append(result)
        time.sleep(2)
        
        # Level 2: Light (vector addition - memory bound)
        print(f"\n[2/5] Light workload (memory-bound)")
        result = self.run_workload_with_monitoring(
            lambda: self.vector_add_workload(size=1_000_000, iterations=100),
            "Light (Vector Add)",
            duration_per_level
        )
        results.append(result)
        time.sleep(2)
        
        # Level 3: Medium (small matrix multiply)
        print(f"\n[3/5] Medium workload (compute-bound)")
        result = self.run_workload_with_monitoring(
            lambda: self.matrix_multiply_workload(size=512, iterations=10),
            "Medium (Matrix 512x512)",
            duration_per_level
        )
        results.append(result)
        time.sleep(2)
        
        # Level 4: High (large matrix multiply)
        print(f"\n[4/5] High workload (compute-intensive)")
        result = self.run_workload_with_monitoring(
            lambda: self.matrix_multiply_workload(size=1024, iterations=5),
            "High (Matrix 1024x1024)",
            duration_per_level
        )
        results.append(result)
        time.sleep(2)
        
        # Level 5: Maximum (Mandelbrot - complex arithmetic)
        print(f"\n[5/5] Maximum workload (complex compute)")
        result = self.run_workload_with_monitoring(
            lambda: self.mandelbrot_workload(width=2048, height=2048, max_iter=1000, iterations=5),
            "Maximum (Mandelbrot 2048x2048)",
            duration_per_level
        )
        results.append(result)
        
        return results


def generate_markdown_report(results: List[Dict]) -> str:
    """Generate markdown report from results."""
    report = "# GPU Power Profiling Results\n\n"
    report += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## Summary Table\n\n"
    report += "| Workload | Mean Power (W) | Min (W) | Max (W) | Std Dev (W) | Energy (J) | Avg Temp (°C) |\n"
    report += "|----------|----------------|---------|---------|-------------|------------|---------------|\n"
    
    for r in results:
        temp_str = f"{r['avg_temperature']:.1f}" if r['avg_temperature'] else "N/A"
        report += f"| {r['workload']} | "
        report += f"{r['mean_power_watts']:.2f} | "
        report += f"{r['min_power_watts']:.2f} | "
        report += f"{r['max_power_watts']:.2f} | "
        report += f"{r['std_power_watts']:.2f} | "
        report += f"{r['total_energy_joules']:.2f} | "
        report += f"{temp_str} |\n"
    
    report += "\n## Analysis\n\n"
    
    # Calculate power range
    idle_power = results[0]['mean_power_watts']
    max_power = max(r['mean_power_watts'] for r in results)
    power_range = max_power - idle_power
    
    report += f"- **Idle Power:** {idle_power:.2f} W\n"
    report += f"- **Maximum Power:** {max_power:.2f} W\n"
    report += f"- **Dynamic Range:** {power_range:.2f} W ({(power_range/idle_power*100):.1f}% increase)\n"
    report += f"- **Sensor Precision:** ±0.01 W (kernel hwmon)\n\n"
    
    report += "## Observations\n\n"
    report += "The power profiling framework successfully captured power consumption "
    report += "across different GPU workload intensities using direct kernel sensor access. "
    report += f"The AMD Radeon RX 580 showed a dynamic power range from {idle_power:.2f}W "
    report += f"(idle) to {max_power:.2f}W (maximum compute load).\n\n"
    
    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GPU stress test with power monitoring')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration per intensity level (seconds)')
    parser.add_argument('--output', type=str, default='results/gpu_power_profile.json',
                       help='Output JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed power readings')
    
    args = parser.parse_args()
    
    try:
        # Initialize stress test
        stress_test = GPUStressTest(verbose=args.verbose)
        
        # Run intensity sweep
        results = stress_test.run_intensity_sweep(duration_per_level=args.duration)
        
        # Save results
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'gpu_info': {
                'name': stress_test.device.name,
                'vendor': stress_test.device.vendor,
                'max_compute_units': stress_test.device.max_compute_units,
                'max_work_group_size': stress_test.device.max_work_group_size
            },
            'power_monitoring': {
                'method': stress_test.power_monitor.method,
                'sensor_path': stress_test.power_monitor.hwmon_path if hasattr(stress_test.power_monitor, 'hwmon_path') else None
            },
            'results': results
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results saved to: {args.output}")
        
        # Generate and save markdown report
        markdown_report = generate_markdown_report(results)
        md_output = args.output.replace('.json', '.md')
        with open(md_output, 'w') as f:
            f.write(markdown_report)
        
        print(f"✅ Report saved to: {md_output}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(markdown_report)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
