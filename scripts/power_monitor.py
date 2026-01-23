#!/usr/bin/env python3
"""
Real-time GPU Power Monitoring - AMD RX 580
============================================

Monitors actual GPU power consumption using kernel sensors.
Supports both direct sensor reading and rocm-smi fallback.

Usage:
    # Monitor power for 60 seconds
    python scripts/power_monitor.py --duration 60
    
    # Monitor with plotting
    python scripts/power_monitor.py --duration 30 --plot
    
    # Use as library
    from scripts.power_monitor import GPUPowerMonitor
    monitor = GPUPowerMonitor()
    power = monitor.read_power()
"""

import time
import glob
import subprocess
import re
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json
import sys


@dataclass
class PowerReading:
    """Single power measurement."""
    timestamp: float
    power_watts: float
    voltage: Optional[float] = None
    temperature: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PowerStatistics:
    """Statistical summary of power readings."""
    duration_seconds: float
    num_samples: int
    mean_power: float
    min_power: float
    max_power: float
    std_power: float
    total_energy_joules: float
    avg_temperature: Optional[float] = None
    
    def __str__(self):
        s = f"Power Statistics ({self.duration_seconds:.1f}s, {self.num_samples} samples)\n"
        s += "=" * 60 + "\n"
        s += f"  Mean Power:        {self.mean_power:.2f} W\n"
        s += f"  Min Power:         {self.min_power:.2f} W\n"
        s += f"  Max Power:         {self.max_power:.2f} W\n"
        s += f"  Std Dev:           {self.std_power:.2f} W\n"
        s += f"  Total Energy:      {self.total_energy_joules:.2f} J ({self.total_energy_joules/3600:.4f} Wh)\n"
        if self.avg_temperature:
            s += f"  Avg Temperature:   {self.avg_temperature:.1f} °C\n"
        return s


class GPUPowerMonitor:
    """
    Monitor AMD GPU power consumption using kernel sensors.
    
    Reads from /sys/class/hwmon/hwmonX/ where X is the AMD GPU device.
    Falls back to rocm-smi or simulation mode if direct reading fails.
    """
    
    def __init__(self, verbose: bool = False, allow_simulation: bool = True):
        """
        Initialize power monitor.
        
        Args:
            verbose: Print debug information
            allow_simulation: Allow simulated power readings if hardware unavailable
        """
        self.verbose = verbose
        self.hwmon_path = None
        self.method = None
        self.base_power = 45.0  # Idle power estimate (W)
        self.tdp = 185.0  # RX 580 TDP
        self.power_file_name = 'power1_average'  # Default, can be power1_input
        
        # Check for simulation mode via environment variable
        simulate_env = os.environ.get('POWER_SIMULATE', '0').lower() in ('1', 'true', 'yes')
        
        # Try to find AMD GPU hwmon device
        try:
            self.hwmon_path = self._find_amdgpu_hwmon()
            self.method = 'kernel_sensors'
            if self.verbose:
                print(f"✅ Found AMD GPU at: {self.hwmon_path}")
        except RuntimeError as e:
            if self.verbose:
                print(f"⚠️  Kernel sensors not available: {e}")
            
            # Try to find any amdgpu hwmon (even without power sensor)
            amdgpu_path = self._find_any_amdgpu_hwmon()
            if amdgpu_path:
                self.hwmon_path = amdgpu_path
                self.method = 'estimated'
                if self.verbose:
                    print(f"✅ Found AMD GPU (no power sensor): {self.hwmon_path}")
                    print("   Using temperature-based power estimation")
            # Try rocm-smi fallback
            elif self._check_rocm_smi():
                self.method = 'rocm_smi'
                if self.verbose:
                    print("✅ Using rocm-smi for power monitoring")
            # Use simulation mode
            elif simulate_env or allow_simulation:
                self.method = 'simulated'
                if self.verbose:
                    print("⚠️  Using SIMULATED power readings")
                    print("   Set POWER_SIMULATE=0 to disable simulation")
            else:
                raise RuntimeError(
                    "Neither kernel sensors nor rocm-smi available for power monitoring. "
                    "Make sure AMD GPU drivers are installed or enable simulation mode."
                )
    
    def _find_amdgpu_hwmon(self) -> str:
        """
        Find AMD GPU in /sys/class/hwmon/ with power sensor.
        
        Returns:
            Path to hwmon directory
            
        Raises:
            RuntimeError if not found
        """
        # Try multiple power sensor filenames
        power_files = ['power1_average', 'power1_input']
        
        for hwmon in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
            name_file = Path(hwmon) / 'name'
            if name_file.exists():
                try:
                    with open(name_file) as f:
                        name = f.read().strip()
                        if 'amdgpu' in name.lower():
                            # Verify power sensor exists (try multiple names)
                            for power_name in power_files:
                                power_file = Path(hwmon) / power_name
                                if power_file.exists():
                                    self.power_file_name = power_name
                                    return hwmon
                except (IOError, PermissionError):
                    continue
        
        raise RuntimeError("AMD GPU hwmon not found in /sys/class/hwmon/")
    
    def _find_any_amdgpu_hwmon(self) -> Optional[str]:
        """
        Find any AMD GPU in /sys/class/hwmon/ (even without power sensor).
        
        Returns:
            Path to hwmon directory or None
        """
        for hwmon in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
            name_file = Path(hwmon) / 'name'
            if name_file.exists():
                try:
                    with open(name_file) as f:
                        name = f.read().strip()
                        if 'amdgpu' in name.lower() or 'radeon' in name.lower():
                            return hwmon
                except (IOError, PermissionError):
                    continue
        return None
    
    def _check_rocm_smi(self) -> bool:
        """Check if rocm-smi is available."""
        try:
            subprocess.check_output(
                ['rocm-smi', '--help'],
                stderr=subprocess.DEVNULL
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def read_power(self) -> float:
        """
        Read current GPU power consumption.
        
        Returns:
            Power in watts
        """
        if self.method == 'kernel_sensors':
            return self._read_power_kernel()
        elif self.method == 'rocm_smi':
            return self._read_power_rocm_smi()
        elif self.method == 'estimated':
            return self._estimate_power_from_temperature()
        elif self.method == 'simulated':
            return self._simulate_power()
        return 0.0
    
    def _read_power_kernel(self) -> float:
        """Read power from kernel sensor."""
        power_file = Path(self.hwmon_path) / self.power_file_name
        try:
            with open(power_file) as f:
                # File returns microwatts
                microwatts = int(f.read().strip())
                return microwatts / 1_000_000
        except (IOError, ValueError, PermissionError) as e:
            if self.verbose:
                print(f"Warning: Failed to read power: {e}")
            return 0.0
    
    def _read_power_rocm_smi(self) -> float:
        """Read power from rocm-smi."""
        try:
            output = subprocess.check_output(
                ['rocm-smi', '--showpower'],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            # Parse: "GPU[0]  : Average Graphics Package Power: 85.0 W"
            match = re.search(r'(\d+\.?\d*)\s*W', output)
            if match:
                return float(match.group(1))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return 0.0
    
    def _estimate_power_from_temperature(self) -> float:
        """
        Estimate power from GPU temperature.
        
        This is a rough approximation based on thermal characteristics.
        Idle temp ~30-40°C = ~45W, Max temp ~85°C = ~185W (TDP)
        
        Returns:
            Estimated power in watts
        """
        temp = self.read_temperature()
        if temp is None:
            return self.base_power
        
        # Linear interpolation between idle and TDP
        # Assuming: 35°C idle (45W), 85°C full load (185W TDP)
        temp_idle = 35.0
        temp_max = 85.0
        
        if temp <= temp_idle:
            return self.base_power
        elif temp >= temp_max:
            return self.tdp
        else:
            # Linear interpolation
            ratio = (temp - temp_idle) / (temp_max - temp_idle)
            return self.base_power + ratio * (self.tdp - self.base_power)
    
    def _simulate_power(self) -> float:
        """
        Simulate power readings for development/testing.
        
        Generates realistic power values with noise.
        
        Returns:
            Simulated power in watts
        """
        import random
        
        # Simulate varying load: 60-120W with noise
        base = 85.0
        variation = 25.0
        noise = 5.0
        
        return base + random.uniform(-variation, variation) + random.gauss(0, noise)
    
    def read_voltage(self) -> Optional[float]:
        """Read GPU voltage in volts (kernel sensors only)."""
        if self.method not in ('kernel_sensors', 'estimated'):
            return None
        
        if not self.hwmon_path:
            return None
        
        voltage_file = Path(self.hwmon_path) / 'in0_input'
        try:
            with open(voltage_file) as f:
                millivolts = int(f.read().strip())
                return millivolts / 1000
        except (IOError, ValueError, PermissionError):
            return None
    
    def read_temperature(self) -> Optional[float]:
        """Read GPU temperature in Celsius."""
        if self.method not in ('kernel_sensors', 'estimated'):
            return None
        
        if not self.hwmon_path:
            return None
        
        temp_file = Path(self.hwmon_path) / 'temp1_input'
        try:
            with open(temp_file) as f:
                millicelsius = int(f.read().strip())
                return millicelsius / 1000
        except (IOError, ValueError, PermissionError):
            return None
    
    def read_full(self) -> PowerReading:
        """Read all available metrics."""
        return PowerReading(
            timestamp=time.time(),
            power_watts=self.read_power(),
            voltage=self.read_voltage(),
            temperature=self.read_temperature()
        )
    
    def monitor_continuous(
        self, 
        duration: float = 60, 
        interval: float = 0.1,
        callback: Optional[callable] = None
    ) -> List[PowerReading]:
        """
        Monitor power continuously.
        
        Args:
            duration: Duration in seconds
            interval: Sampling interval in seconds
            callback: Optional callback function called on each reading
            
        Returns:
            List of power readings
        """
        readings = []
        start = time.time()
        
        print(f"📊 Monitoring power for {duration:.0f} seconds (interval: {interval:.2f}s)...")
        
        while time.time() - start < duration:
            reading = self.read_full()
            readings.append(reading)
            
            if callback:
                callback(reading)
            
            time.sleep(interval)
        
        elapsed = time.time() - start
        print(f"✅ Captured {len(readings)} samples in {elapsed:.2f}s")
        
        return readings
    
    def calculate_statistics(self, readings: List[PowerReading]) -> PowerStatistics:
        """Calculate statistics from readings."""
        if not readings:
            raise ValueError("No readings to analyze")
        
        powers = [r.power_watts for r in readings]
        temps = [r.temperature for r in readings if r.temperature is not None]
        
        duration = readings[-1].timestamp - readings[0].timestamp
        
        import statistics
        
        # Energy calculation (trapezoidal rule for better accuracy)
        energy = 0.0
        for i in range(len(readings) - 1):
            dt = readings[i+1].timestamp - readings[i].timestamp
            avg_power = (readings[i].power_watts + readings[i+1].power_watts) / 2
            energy += avg_power * dt
        
        return PowerStatistics(
            duration_seconds=duration,
            num_samples=len(readings),
            mean_power=statistics.mean(powers),
            min_power=min(powers),
            max_power=max(powers),
            std_power=statistics.stdev(powers) if len(powers) > 1 else 0.0,
            total_energy_joules=energy,
            avg_temperature=statistics.mean(temps) if temps else None
        )
    
    def save_readings(self, readings: List[PowerReading], filename: str):
        """Save readings to JSON file."""
        data = {
            'method': self.method,
            'readings': [r.to_dict() for r in readings],
            'statistics': asdict(self.calculate_statistics(readings))
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(readings)} readings to {filename}")


def plot_power_readings(readings: List[PowerReading], title: str = "GPU Power Consumption"):
    """
    Plot power readings (requires matplotlib).
    
    Args:
        readings: List of power readings
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return
    
    times = [(r.timestamp - readings[0].timestamp) for r in readings]
    powers = [r.power_watts for r in readings]
    
    plt.figure(figsize=(12, 6))
    
    # Power plot
    plt.subplot(2, 1, 1)
    plt.plot(times, powers, linewidth=0.5, alpha=0.7)
    plt.ylabel('Power (W)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Statistics
    import statistics
    mean_power = statistics.mean(powers)
    plt.axhline(y=mean_power, color='r', linestyle='--', label=f'Mean: {mean_power:.1f}W')
    plt.legend()
    
    # Temperature plot (if available)
    temps = [r.temperature for r in readings if r.temperature is not None]
    if temps:
        plt.subplot(2, 1, 2)
        plt.plot(times[:len(temps)], temps, color='orange', linewidth=0.5, alpha=0.7)
        plt.ylabel('Temperature (°C)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    else:
        plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('power_monitoring.png', dpi=150)
    print("📊 Plot saved to power_monitoring.png")
    plt.show()


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor AMD GPU power consumption')
    parser.add_argument('--duration', type=float, default=60, help='Monitoring duration (seconds)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval (seconds)')
    parser.add_argument('--output', type=str, help='Save readings to JSON file')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = GPUPowerMonitor(verbose=args.verbose)
        
        print(f"\n{'='*60}")
        print(f"AMD GPU Power Monitor")
        print(f"{'='*60}")
        print(f"Method: {monitor.method}")
        print(f"Duration: {args.duration}s")
        print(f"Interval: {args.interval}s")
        print(f"{'='*60}\n")
        
        # Initial reading
        initial = monitor.read_full()
        print(f"Current power: {initial.power_watts:.2f} W")
        if initial.temperature:
            print(f"Current temp:  {initial.temperature:.1f} °C")
        print()
        
        # Monitor continuously
        def print_callback(reading):
            """Print live updates."""
            sys.stdout.write(f"\r⚡ {reading.power_watts:6.2f} W")
            if reading.temperature:
                sys.stdout.write(f"  |  🌡️  {reading.temperature:5.1f} °C")
            sys.stdout.flush()
        
        readings = monitor.monitor_continuous(
            duration=args.duration,
            interval=args.interval,
            callback=print_callback
        )
        print()  # Newline after progress
        
        # Calculate and print statistics
        stats = monitor.calculate_statistics(readings)
        print(f"\n{stats}")
        
        # Save if requested
        if args.output:
            monitor.save_readings(readings, args.output)
        
        # Plot if requested
        if args.plot:
            plot_power_readings(readings)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
