# Radeon RX 580 AI Framework - Example Usage

Welcome! Here's a quick example to get started.

## Example 1: Simple Hardware Check

```python
from src.core.gpu import GPUManager
from src.core.memory import MemoryManager

# Initialize GPU
gpu = GPUManager()
if gpu.initialize():
    print("GPU ready!")
    info = gpu.get_info()
    print(f"Using: {info.name}")
    print(f"Backend: {gpu.get_compute_backend()}")

# Check memory
memory = MemoryManager()
memory.print_stats()
```

## Example 2: With Configuration

```python
from src.utils.config import Config, load_config
from src.core.gpu import GPUManager

# Load config
config = load_config('configs/optimized.yaml')

# Initialize with config
gpu = GPUManager()
gpu.initialize()

print(f"Max VRAM: {config.max_vram_usage_mb} MB")
print(f"Quantization: {config.use_quantization}")
```

## Example 3: Performance Profiling

```python
from src.core.profiler import Profiler
import time

profiler = Profiler()

# Profile an operation
profiler.start("my_operation")
time.sleep(0.1)  # Your actual operation here
profiler.end("my_operation")

# See results
profiler.print_summary()
```

## Next Steps

1. Run full diagnostics: `python scripts/diagnostics.py`
2. Try the benchmark: `python scripts/benchmark.py --all`
3. Read the docs: [docs/](docs/)

For more examples, check the `examples/` directory (coming soon).
