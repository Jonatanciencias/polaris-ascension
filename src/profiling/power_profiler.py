"""
Power Profiler for Inference Benchmarks
========================================

Combines power monitoring with inference benchmarking.
"""

import time
import threading
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.power_monitor import GPUPowerMonitor, PowerReading, PowerStatistics


@dataclass
class BenchmarkResults:
    """Results from inference benchmark with power monitoring."""
    # Performance metrics
    duration_seconds: float
    num_inferences: int
    fps: float
    avg_latency_ms: float
    
    # Power metrics
    avg_power_watts: float
    min_power_watts: float
    max_power_watts: float
    total_energy_joules: float
    
    # Efficiency metrics
    energy_per_inference_joules: float
    fps_per_watt: float
    inferences_per_joule: float
    
    # Temperature (if available)
    avg_temperature: Optional[float] = None
    
    def __str__(self):
        s = "Benchmark Results\n"
        s += "=" * 60 + "\n"
        s += "\n📊 Performance:\n"
        s += f"  Duration:          {self.duration_seconds:.2f} s\n"
        s += f"  Inferences:        {self.num_inferences}\n"
        s += f"  FPS:               {self.fps:.2f}\n"
        s += f"  Avg Latency:       {self.avg_latency_ms:.2f} ms\n"
        s += "\n⚡ Power:\n"
        s += f"  Average Power:     {self.avg_power_watts:.2f} W\n"
        s += f"  Min Power:         {self.min_power_watts:.2f} W\n"
        s += f"  Max Power:         {self.max_power_watts:.2f} W\n"
        s += f"  Total Energy:      {self.total_energy_joules:.2f} J ({self.total_energy_joules/3600:.4f} Wh)\n"
        s += "\n💡 Efficiency:\n"
        s += f"  Energy/Inference:  {self.energy_per_inference_joules*1000:.2f} mJ\n"
        s += f"  FPS/Watt:          {self.fps_per_watt:.2f}\n"
        s += f"  Inferences/Joule:  {self.inferences_per_joule:.2f}\n"
        if self.avg_temperature:
            s += f"\n🌡️  Temperature:       {self.avg_temperature:.1f} °C\n"
        return s


class PowerProfiler:
    """
    Profile power consumption during model inference.
    
    Example:
        profiler = PowerProfiler()
        results = profiler.profile_model(model, test_data, duration=60)
        print(results)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize power profiler.
        
        Args:
            verbose: Print debug information
        """
        self.verbose = verbose
        self.power_monitor = GPUPowerMonitor(verbose=verbose)
    
    def profile_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        duration: float = 60,
        warmup_seconds: float = 5
    ) -> BenchmarkResults:
        """
        Profile model inference with power monitoring.
        
        Args:
            model: PyTorch model
            data_loader: DataLoader with test data
            duration: Profiling duration in seconds
            warmup_seconds: Warmup duration before profiling
            
        Returns:
            BenchmarkResults with performance and power metrics
        """
        model.eval()
        device = next(model.parameters()).device
        
        print(f"\n{'='*60}")
        print(f"Power Profiling")
        print(f"{'='*60}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Device: {device}")
        print(f"Duration: {duration}s (+ {warmup_seconds}s warmup)")
        print(f"{'='*60}\n")
        
        # Warmup
        if warmup_seconds > 0:
            print(f"🔥 Warming up for {warmup_seconds}s...")
            self._run_inference(model, data_loader, warmup_seconds)
            print("✅ Warmup complete\n")
        
        # Run profiling with power monitoring
        print(f"📊 Profiling for {duration}s...")
        
        power_readings = []
        inference_times = []
        num_inferences = 0
        stop_monitoring = threading.Event()
        
        def monitor_power():
            """Background thread for power monitoring."""
            while not stop_monitoring.is_set():
                reading = self.power_monitor.read_full()
                power_readings.append(reading)
                time.sleep(0.1)  # 10Hz sampling
        
        # Start power monitoring thread
        monitor_thread = threading.Thread(target=monitor_power, daemon=True)
        monitor_thread.start()
        
        # Run inference
        start_time = time.time()
        
        try:
            with torch.no_grad():
                while time.time() - start_time < duration:
                    for batch in data_loader:
                        if isinstance(batch, (tuple, list)):
                            batch = batch[0]  # Get input tensor
                        
                        batch = batch.to(device)
                        
                        # Measure inference time
                        inf_start = time.time()
                        _ = model(batch)
                        inf_time = time.time() - inf_start
                        
                        inference_times.append(inf_time)
                        num_inferences += batch.size(0)
                        
                        if time.time() - start_time >= duration:
                            break
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join(timeout=2)
        
        elapsed = time.time() - start_time
        
        print(f"✅ Profiling complete\n")
        
        # Calculate statistics
        power_stats = self.power_monitor.calculate_statistics(power_readings)
        
        # Calculate metrics
        fps = num_inferences / elapsed
        avg_latency_ms = (sum(inference_times) / len(inference_times)) * 1000 if inference_times else 0
        energy_per_inference = power_stats.total_energy_joules / num_inferences if num_inferences > 0 else 0
        fps_per_watt = fps / power_stats.mean_power if power_stats.mean_power > 0 else 0
        inferences_per_joule = num_inferences / power_stats.total_energy_joules if power_stats.total_energy_joules > 0 else 0
        
        return BenchmarkResults(
            duration_seconds=elapsed,
            num_inferences=num_inferences,
            fps=fps,
            avg_latency_ms=avg_latency_ms,
            avg_power_watts=power_stats.mean_power,
            min_power_watts=power_stats.min_power,
            max_power_watts=power_stats.max_power,
            total_energy_joules=power_stats.total_energy_joules,
            energy_per_inference_joules=energy_per_inference,
            fps_per_watt=fps_per_watt,
            inferences_per_joule=inferences_per_joule,
            avg_temperature=power_stats.avg_temperature
        )
    
    def _run_inference(self, model: nn.Module, data_loader: DataLoader, duration: float):
        """Run inference for specified duration (used for warmup)."""
        device = next(model.parameters()).device
        start = time.time()
        
        with torch.no_grad():
            while time.time() - start < duration:
                for batch in data_loader:
                    if isinstance(batch, (tuple, list)):
                        batch = batch[0]
                    batch = batch.to(device)
                    _ = model(batch)
                    
                    if time.time() - start >= duration:
                        break


class BenchmarkWithPower:
    """
    Simplified interface for benchmarking with power monitoring.
    
    Example:
        benchmark = BenchmarkWithPower(model, data_loader)
        results = benchmark.run(duration=60)
        print(f"FPS: {results.fps:.1f}")
        print(f"Power: {results.avg_power_watts:.1f}W")
        print(f"FPS/Watt: {results.fps_per_watt:.2f}")
    """
    
    def __init__(self, model: nn.Module, data_loader: DataLoader, verbose: bool = False):
        """
        Initialize benchmark.
        
        Args:
            model: PyTorch model
            data_loader: DataLoader with test data
            verbose: Print debug information
        """
        self.model = model
        self.data_loader = data_loader
        self.profiler = PowerProfiler(verbose=verbose)
    
    def run(
        self,
        duration: float = 60,
        warmup_seconds: float = 5
    ) -> BenchmarkResults:
        """
        Run benchmark.
        
        Args:
            duration: Benchmark duration in seconds
            warmup_seconds: Warmup duration before benchmark
            
        Returns:
            BenchmarkResults
        """
        return self.profiler.profile_model(
            self.model,
            self.data_loader,
            duration=duration,
            warmup_seconds=warmup_seconds
        )
