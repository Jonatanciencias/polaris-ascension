# 🚀 Quick Start Guide

Welcome to the Radeon RX 580 AI Framework! This guide will get you up and running in minutes.

**Version**: 0.3.0 - Now with optimized inference and user-friendly CLI! 🎉

## Prerequisites

- AMD Radeon RX 580 GPU
- Ubuntu 20.04+ (or compatible Linux distribution)
- Python 3.8+
- 16GB+ system RAM recommended

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/radeon-rx580-ai.git
cd radeon-rx580-ai
```

### Step 2: Run Setup Script

```bash
./scripts/setup.sh
```

This script will:
- Install system dependencies (OpenCL, Mesa drivers)
- Create a Python virtual environment
- Install all required Python packages
- Install the framework in development mode

### Step 3: Activate Virtual Environment

```bash
source venv/bin/activate
```

### Step 4: Verify Installation

```bash
python scripts/verify_hardware.py
```

Expected output:
```
============================================================
Radeon RX 580 Hardware Verification
============================================================

[1/3] Detecting GPU...
✅ GPU Detected: AMD Radeon RX 580
   PCI ID: 03:00.0
   Architecture: Polaris 20 (GCN 4.0)
   Driver: amdgpu

[2/3] Checking compute capabilities...
   Recommended backend: OPENCL
   ✅ OpenCL: Available
   ⚠️  ROCm: Not available (optional)

[3/3] Checking memory...
   System RAM: 16.0 GB
   Available RAM: 12.0 GB
   GPU VRAM: 8.0 GB

============================================================
✅ System is ready for AI workloads!
============================================================
```

## Basic Usage

### Option 1: Command-Line Interface (Easiest) **NEW in v0.3.0**

The CLI is the fastest way to get started:

```bash
# Get system information
python -m src.cli info

# Classify a single image (standard quality)
python -m src.cli classify examples/test_images/sample.jpg

# Fast mode (~1.5x speedup with FP16)
python -m src.cli classify examples/test_images/sample.jpg --fast

# Ultra-fast mode (~2.5x speedup with INT8)
python -m src.cli classify examples/test_images/sample.jpg --ultra-fast

# Process multiple images in batch
python -m src.cli classify examples/test_images/*.jpg --batch 4 --fast

# Run performance benchmark
python -m src.cli benchmark
```

**Example Output:**
```
🚀 Initializing Fast Mode (FP16)...
📦 Loading model from examples/models/mobilenetv2.onnx...
✅ Model loaded: mobilenetv2

⚙️  Optimization Settings:
   Precision: FP16
   Batch Size: 1
   Expected Performance: ~1.5x faster than FP32
   Memory Savings: ~50% less memory
   Accuracy: 73.6 dB SNR (safe for medical imaging)

🖼️  Processing 1 image(s)...

📸 sample.jpg
   Top prediction: Class 281 (85.3% confident)
   Top 5 predictions:
      1. Class 281: 85.3%
      2. Class 282: 8.2%
      3. Class 285: 3.1%
      4. Class 246: 1.8%
      5. Class 340: 0.9%

📈 Performance Statistics:
   Average Inference Time: 340.5ms
   Throughput: 2.9 images/second
```

### Option 2: Python API (For Developers)

```python
from core.gpu import GPUManager

# Initialize GPU manager
gpu = GPUManager()
if gpu.initialize():
    info = gpu.get_info()
    print(f"GPU: {info.name}")
    print(f"VRAM: {info.vram_mb} MB")
    print(f"Backend: {gpu.get_compute_backend()}")
```

### Example 2: Memory Management

#### Example 4: Memory Management

```python
from src.core.memory import MemoryManager

# Create memory manager
memory = MemoryManager(gpu_vram_mb=8192)

# Check if we can allocate 4GB for a model
if memory.can_allocate(4096, use_gpu=True):
    print("✅ Model will fit in VRAM")
    memory.register_allocation("my_model", 4096, is_gpu=True)
    memory.print_stats()
```

#### Example 5: Performance Profiling

#### Example 5: Performance Profiling

```python
from src.core.profiler import Profiler
import time

profiler = Profiler()

# Profile an operation
profiler.start("inference")
time.sleep(0.5)  # Your inference code here
profiler.end("inference")

# Print results
profiler.print_summary()
```

#### Example 6: Load Configuration

```python
from utils.config import load_config

# Load optimized configuration
config = load_config('configs/optimized.yaml')

print(f"GPU Backend: {config.gpu_backend}")
print(f"Max VRAM: {config.max_vram_usage_mb} MB")
print(f"Quantization: {config.use_quantization}")
print(f"Batch Size: {config.default_batch_size}")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gpu.py -v

# Run with coverage
pytest tests/ --cov=core --cov=utils
```

## Running Diagnostics

```bash
# Full system diagnostics
python scripts/diagnostics.py

# Hardware benchmarks
python scripts/benchmark.py --all
```

## Configuration

### Default Configuration

Located at `configs/default.yaml`:

```yaml
gpu_backend: auto
max_vram_usage_mb: 7168
enable_cpu_offload: true
use_quantization: false
default_batch_size: 1
```

### Optimized Configuration

Located at `configs/optimized.yaml`:

```yaml
gpu_backend: opencl
max_vram_usage_mb: 6656
enable_cpu_offload: true
use_quantization: true
quantization_bits: 8
default_batch_size: 1
```

## Troubleshooting

### OpenCL Not Available

```bash
# Install OpenCL
sudo apt install ocl-icd-opencl-dev opencl-headers clinfo

# Verify installation
clinfo --list
```

### GPU Not Detected

```bash
# Check if GPU is visible
lspci | grep -i vga

# Check driver
lsmod | grep amdgpu

# If driver not loaded
sudo modprobe amdgpu
```

### Permission Issues

```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

## Next Steps

1. **Read the documentation**:
   - [Architecture](docs/architecture.md)
   - [Optimization Guide](docs/optimization.md)
   - [Contributing](docs/contributing.md)

2. **Try examples**:
   - See `examples/README.md`

3. **Run inference** (coming soon):
   - Stable Diffusion implementation
   - Other AI models

4. **Optimize performance**:
   - Profile your workload
   - Apply quantization
   - Tune batch sizes

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Verify hardware
python scripts/verify_hardware.py

# Run diagnostics
python scripts/diagnostics.py

# Run benchmarks
python scripts/benchmark.py --memory
python scripts/benchmark.py --compute
python scripts/benchmark.py --all

# Run tests
pytest tests/ -v

# Format code (for contributors)
black src/ tests/
flake8 src/ tests/

# Deactivate environment
deactivate
```

## Getting Help

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check the `docs/` directory

## Project Status

**Current Phase**: Foundation (v0.1.0-alpha)
- ✅ Hardware detection and verification
- ✅ Memory management system
- ✅ Performance profiling tools
- ✅ Configuration management
- ✅ Comprehensive testing

**Next Phase**: Core Inference
- ⏳ PyTorch/ONNX integration
- ⏳ Stable Diffusion implementation
- ⏳ Model optimization pipeline

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

---

**Happy coding!** 🚀

For detailed documentation, visit the [docs/](docs/) directory.
