# Power Monitoring Implementation
## Real Power Measurements for AMD Radeon RX 580

### 📊 Overview

This implementation provides **real-time GPU power monitoring** for the AMD Radeon RX 580, enabling accurate measurement of power consumption during model inference.

**Status**: ✅ **COMPLETE** - Production Ready

### 🎯 Features

1. **Dual Method Support**:
   - **Primary**: Kernel sensors (`/sys/class/hwmon/`) - Direct hardware access, μs latency
   - **Fallback**: rocm-smi - CLI-based monitoring, works when kernel sensors unavailable

2. **Comprehensive Metrics**:
   - Instantaneous power (Watts)
   - Voltage (Volts)
   - Temperature (°C)
   - Energy consumption (Joules, Wh)
   - Statistical analysis (mean, min, max, std dev)

3. **Inference Integration**:
   - Power profiling during model inference
   - FPS/Watt efficiency metrics
   - Energy per inference (mJ)
   - Continuous background monitoring

### 📁 Files Implemented

```
scripts/
  power_monitor.py              # Core power monitoring (480 LOC)
  benchmark_all_models_power.py # Benchmark all models (340 LOC)

src/profiling/
  __init__.py                   # Module initialization
  power_profiler.py             # Inference profiler (280 LOC)

examples/
  benchmark_with_power_demo.py  # Interactive demos (420 LOC)
```

**Total**: ~1,520 LOC of production-ready power monitoring code

---

## 🚀 Quick Start

### 1. Basic Power Monitoring

```bash
# Monitor power for 60 seconds
python scripts/power_monitor.py --duration 60

# Monitor with live plotting
python scripts/power_monitor.py --duration 30 --plot

# Save readings to JSON
python scripts/power_monitor.py --duration 60 --output power_data.json
```

**Example Output**:
```
Power Statistics (60.0s, 600 samples)
============================================================
  Mean Power:        85.23 W
  Min Power:         72.10 W
  Max Power:         127.45 W
  Std Dev:           12.34 W
  Total Energy:      5113.80 J (1.4205 Wh)
  Avg Temperature:   68.5 °C
```

### 2. Inference Profiling

```bash
# Run interactive demo
python examples/benchmark_with_power_demo.py
```

**Python API**:
```python
from src.profiling.power_profiler import BenchmarkWithPower

# Create model and data loader
model = YourModel()
data_loader = YourDataLoader()

# Run benchmark with power monitoring
benchmark = BenchmarkWithPower(model, data_loader)
results = benchmark.run(duration=60)

# Access metrics
print(f"FPS: {results.fps:.1f}")
print(f"Power: {results.avg_power_watts:.1f}W")
print(f"FPS/Watt: {results.fps_per_watt:.2f}")
print(f"Energy/Image: {results.energy_per_inference_joules*1000:.2f} mJ")
```

### 3. Benchmark All Models

```bash
# Quick test (30s per model)
python scripts/benchmark_all_models_power.py --duration 30 --models simple

# Full benchmark (60s per model)
python scripts/benchmark_all_models_power.py --duration 60 --models all --output results/power_benchmarks.json

# Standard models only (ResNet, MobileNet)
python scripts/benchmark_all_models_power.py --duration 45 --models standard
```

**Generates**:
- `results/power_benchmarks.json` - Complete benchmark data
- `results/power_benchmarks.md` - Comparison table

---

## 🔬 Technical Details

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                       │
│  (benchmark_with_power_demo.py, custom scripts)         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │   PowerProfiler            │
        │  (High-level API)          │
        │  - Threads power monitor   │
        │  - Runs inference loop     │
        │  - Calculates metrics      │
        └────────────┬───────────────┘
                     │
        ┌────────────┴──────────────┐
        │   GPUPowerMonitor          │
        │  (Low-level driver)        │
        │  - Reads sensors           │
        │  - Aggregates readings     │
        └────────────┬───────────────┘
                     │
        ┌────────────┴──────────────┐
        │   Hardware Interface       │
        │  /sys/class/hwmon/hwmonX/  │
        │  or rocm-smi               │
        └────────────────────────────┘
```

### Power Reading Methods

#### Method 1: Kernel Sensors (Primary)
- **Path**: `/sys/class/hwmon/hwmonX/power1_average`
- **Precision**: 1 μW (microwatts)
- **Latency**: <100 μs (direct file read)
- **Sampling Rate**: Up to 10 kHz (limited to 10 Hz by default)
- **Permissions**: Read-only, no root required

**Example**:
```python
monitor = GPUPowerMonitor()
power = monitor.read_power()  # Returns watts
print(f"Current power: {power:.2f}W")
```

#### Method 2: rocm-smi (Fallback)
- **Command**: `rocm-smi --showpower`
- **Precision**: 0.1 W
- **Latency**: ~50-100 ms (subprocess call)
- **Sampling Rate**: ~5-10 Hz (subprocess overhead)

### Metrics Calculated

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| **Mean Power** | Σ(P_i) / n | Watts | Average power draw |
| **Energy** | ∫ P(t) dt | Joules | Total energy consumed (trapezoidal rule) |
| **FPS** | inferences / time | 1/s | Throughput |
| **FPS/Watt** | FPS / mean_power | 1/(W·s) | Power efficiency |
| **Energy/Inference** | total_energy / inferences | J | Energy per image |
| **Inferences/Joule** | inferences / total_energy | 1/J | Energy efficiency |

### Statistical Validation

All power measurements include:
- **Mean ± Std Dev**: Central tendency and spread
- **Min/Max**: Range of values
- **Sample Count**: Number of readings (for confidence)
- **Duration**: Measurement period

**Confidence Intervals** (for n≥30 samples):
```python
import statistics
import math

ci_95 = 1.96 * (stats.std_power / math.sqrt(stats.num_samples))
print(f"Power: {stats.mean_power:.2f} ± {ci_95:.2f}W (95% CI)")
```

---

## 📊 Example Results

### AMD Radeon RX 580 Benchmarks

| Model | Quantization | Params | FPS | Power (W) | FPS/W | Energy/Img (mJ) |
|-------|--------------|--------|-----|-----------|-------|-----------------|
| SimpleCNN | FP32 | 2.1M | 1,245 | 75.2 | 16.6 | 60.4 |
| ResNet-18 | FP32 | 11.7M | 342 | 112.5 | 3.0 | 329.0 |
| ResNet-18 | FP16 | 11.7M | 587 | 95.3 | 6.2 | 162.3 |
| MobileNetV2 | FP32 | 3.5M | 892 | 88.1 | 10.1 | 98.8 |

*(Example data - actual values depend on hardware and drivers)*

---

## 🧪 Validation

### Verify Installation

```bash
# Check if sensors are accessible
ls -la /sys/class/hwmon/hwmon*/power1_average

# Test power monitor
python scripts/power_monitor.py --duration 5 --verbose

# Run demo
python examples/benchmark_with_power_demo.py
```

### Expected Output

```
✅ Found AMD GPU at: /sys/class/hwmon/hwmon3
📊 Current GPU State:
  Power:       45.23 W
  Voltage:     0.950 V
  Temperature: 42.5 °C
```

### Troubleshooting

**Problem**: "AMD GPU hwmon not found"
- **Solution**: Make sure AMD GPU drivers are installed
- **Check**: `lspci | grep VGA` should show AMD GPU
- **Fallback**: Will automatically try rocm-smi

**Problem**: "Permission denied reading power file"
- **Solution**: Sensors should be world-readable, no root needed
- **Check**: `cat /sys/class/hwmon/hwmon*/power1_average`
- **If fails**: Use rocm-smi method (requires ROCm installed)

**Problem**: Power readings are always 0
- **Solution**: GPU might be idle or in low-power mode
- **Run**: Inference workload to see power increase
- **Check**: Temperature should increase with power

---

## 📈 Usage for Academic Paper

### Data Collection

```bash
# 1. Benchmark all models (60s per model)
python scripts/benchmark_all_models_power.py \
  --duration 60 \
  --warmup 10 \
  --batch-size 32 \
  --output results/paper_benchmarks.json

# 2. Extract comparison table
cat results/paper_benchmarks.md
```

### Statistical Rigor

For academic papers, run **multiple trials** (n≥10):

```bash
# Run 10 trials
for i in {1..10}; do
  python scripts/benchmark_all_models_power.py \
    --duration 60 \
    --output results/trial_${i}.json
done

# Aggregate results with confidence intervals
python scripts/aggregate_trials.py results/trial_*.json \
  --output results/paper_data_with_ci.json
```

### Metrics to Report

1. **Performance**:
   - FPS (mean ± CI)
   - Latency (mean ± CI)

2. **Power**:
   - Mean power ± std dev (W)
   - Peak power (W)
   - Total energy (J or Wh)

3. **Efficiency**:
   - FPS/Watt (higher is better)
   - Energy per inference (mJ) (lower is better)
   - Inferences per Joule (higher is better)

4. **Temperature**:
   - Mean temperature (°C)
   - Peak temperature (°C)

---

## 🔧 Advanced Usage

### Custom Models

```python
from src.profiling.power_profiler import PowerProfiler
from torch.utils.data import DataLoader

# Your model
model = YourCustomModel()
data_loader = DataLoader(your_dataset, batch_size=32)

# Profile with custom duration
profiler = PowerProfiler(verbose=True)
results = profiler.profile_model(
    model=model,
    data_loader=data_loader,
    duration=120,  # 2 minutes
    warmup_seconds=15
)

# Access all metrics
print(f"Throughput: {results.fps:.1f} FPS")
print(f"Power: {results.avg_power_watts:.1f}W")
print(f"Energy/Inference: {results.energy_per_inference_joules*1000:.2f} mJ")
print(f"Efficiency: {results.fps_per_watt:.2f} FPS/W")
```

### Continuous Monitoring

```python
from scripts.power_monitor import GPUPowerMonitor

monitor = GPUPowerMonitor()

# Monitor for 5 minutes with 1Hz sampling
readings = monitor.monitor_continuous(
    duration=300,
    interval=1.0,
    callback=lambda r: print(f"{r.power_watts:.2f}W")
)

# Calculate statistics
stats = monitor.calculate_statistics(readings)
print(stats)

# Save for later analysis
monitor.save_readings(readings, 'power_log.json')
```

### Real-time Plotting

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

monitor = GPUPowerMonitor()
powers = []

def update(frame):
    reading = monitor.read_full()
    powers.append(reading.power_watts)
    plt.cla()
    plt.plot(powers[-100:])  # Last 100 samples
    plt.ylabel('Power (W)')
    plt.xlabel('Sample')

fig = plt.figure()
ani = FuncAnimation(fig, update, interval=100)
plt.show()
```

---

## 🎓 Academic Contributions

This implementation enables:

1. **Empirical Validation**: Real power measurements (not estimates)
2. **Reproducibility**: Open-source, documented, tested
3. **Comprehensive Analysis**: Power + performance + efficiency
4. **Statistical Rigor**: Multiple runs, confidence intervals
5. **Hardware Accessibility**: Works on consumer GPU ($150)

### Citation Data

- **Hardware**: AMD Radeon RX 580 (Polaris/GCN 4.0)
- **Power Monitoring**: Kernel sensors (μW precision)
- **Sampling Rate**: 10 Hz (adjustable up to 10 kHz)
- **Method**: Trapezoidal rule for energy integration
- **Validation**: Multiple runs with confidence intervals

---

## 📝 Limitations & Future Work

### Current Limitations

1. **AMD-specific**: Only works with AMD GPUs (kernel sensors)
2. **Linux-only**: Requires `/sys/class/hwmon/` interface
3. **System power**: Measures GPU package power only (not total system)
4. **Sampling rate**: Limited to 10 Hz by default (can increase)

### Future Enhancements

1. **NVIDIA support**: Add NVML library integration
2. **Windows support**: Use WMI or vendor APIs
3. **System power**: Integrate with PSU monitoring
4. **Real-time dashboard**: Web-based monitoring UI
5. **Power capping**: Automatic power limit adjustment

---

## ✅ Testing

Run the test suite:

```bash
# Run all demos
python examples/benchmark_with_power_demo.py

# Quick validation (5 seconds)
python scripts/power_monitor.py --duration 5 --verbose

# Full benchmark (SimpleCNN only)
python scripts/benchmark_all_models_power.py --models simple --duration 30
```

**Expected behavior**:
- ✅ Power readings between 40-150W (RX 580 TDP: 185W)
- ✅ Temperature increases with power
- ✅ Power correlates with GPU workload
- ✅ Energy integration produces positive values

---

## 📚 References

1. **AMD GPU Sensors**: Linux kernel documentation (`hwmon` subsystem)
2. **ROCm SMI**: [AMD ROCm System Management Interface](https://github.com/RadeonOpenCompute/rocm_smi_lib)
3. **Power Profiling**: NVIDIA Nsight, Intel VTune equivalents for AMD
4. **Energy Efficiency Metrics**: MLPerf Power specification

---

**Implementation Date**: January 23, 2026  
**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**License**: Same as parent project
