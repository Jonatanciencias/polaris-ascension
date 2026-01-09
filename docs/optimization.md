# Optimization Techniques for RX 580

This document describes optimization strategies for running AI workloads on AMD Radeon RX 580 GPUs.

## Memory Optimization

### 1. Model Quantization

#### 8-bit Quantization

Reduces model size by ~50% with minimal quality loss.

```python
from src.utils.config import Config

config = Config(
    use_quantization=True,
    quantization_bits=8
)
```

**Benefits:**
- 50% memory reduction
- Faster inference (less data transfer)
- Minimal accuracy loss (<2%)

**Trade-offs:**
- Slight quality degradation
- Initial quantization overhead

#### 4-bit Quantization

Aggressive quantization for very large models.

```python
config = Config(
    use_quantization=True,
    quantization_bits=4
)
```

**Benefits:**
- 75% memory reduction
- Fits larger models in VRAM

**Trade-offs:**
- More quality degradation (5-10%)
- May require calibration

### 2. CPU Offloading

Offload parts of the model to system RAM.

```python
config = Config(
    enable_cpu_offload=True,
    offload_threshold_mb=512
)
```

**Strategy:**
- Keep frequently-used layers in VRAM
- Offload encoder/decoder to RAM
- Stream weights during inference

**Best for:**
- Models slightly larger than VRAM
- Sequential architectures

### 3. Gradient Checkpointing

Trade compute for memory during inference.

```python
# Coming soon: Enable gradient checkpointing
# Recompute activations instead of storing them
```

### 4. Batch Size Optimization

Find optimal batch size for your workload.

```python
from src.core.memory import MemoryManager

memory = MemoryManager()
recs = memory.get_recommendations(model_size_mb=3500)
print(f"Recommended batch size: {recs['max_batch_size']}")
```

## Compute Optimization

### 1. Kernel Fusion

Combine operations to reduce kernel launches.

```python
# Example: Fuse activation and normalization
# Instead of: norm(relu(x))
# Use: fused_norm_relu(x)
```

### 2. Memory Access Patterns

Optimize memory access for GPU architecture.

**Coalesced Access:**
```c
// Good: Sequential access
for (int i = 0; i < n; i++) {
    output[i] = input[i] * 2.0f;
}

// Bad: Strided access
for (int i = 0; i < n; i++) {
    output[i] = input[i * stride] * 2.0f;
}
```

### 3. Work Group Optimization

Tune OpenCL work group sizes.

```python
# Optimal for Polaris (GCN 4.0):
work_group_size = 256  # Multiple of 64 (wavefront size)
```

### 4. Precision Selection

Use appropriate precision for each operation.

```python
# FP16 for most operations (if supported)
# FP32 for critical operations (e.g., normalization)
# INT8 for quantized operations
```

## Model-Specific Optimizations

### Stable Diffusion

#### Resolution Optimization

```python
# Start with lower resolutions
resolution = 512  # Instead of 768 or 1024

# Use tiling for larger images
use_tiling = True
tile_size = 512
```

#### Step Reduction

```python
# Reduce inference steps
num_inference_steps = 15  # Instead of 50

# Use better schedulers (DPM++, DDIM)
scheduler = "DPMSolverMultistep"
```

#### Model Variants

```python
# Use smaller model variants
model = "stabilityai/stable-diffusion-2-1-base"  # Not XL
# or distilled models
model = "nota-ai/bk-sdm-small"  # Distilled version
```

### Text Models (Future)

```python
# Context length limits
max_context_length = 512  # Instead of 2048

# Use smaller models
# GPT-2 instead of GPT-3
# DistilBERT instead of BERT
```

## Backend Optimization

### OpenCL

#### Platform Selection

```python
# Prefer AMD platform
platform = "AMD Accelerated Parallel Processing"

# Use newest OpenCL version available
opencl_version = "2.0"
```

#### Compiler Flags

```python
# Optimize compilation
build_options = [
    "-cl-fast-relaxed-math",
    "-cl-mad-enable",
    "-cl-no-signed-zeros"
]
```

### ROCm (If Available)

```python
# Enable TensorCore-like operations
# Set environment variables
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
```

## Profiling and Monitoring

### Performance Profiling

```python
from src.core.profiler import Profiler

profiler = Profiler()

profiler.start("inference")
# ... run inference ...
profiler.end("inference")

profiler.print_summary()
```

### Memory Profiling

```python
from src.core.memory import MemoryManager

memory = MemoryManager()

# Before inference
memory.print_stats()

# Register allocations
memory.register_allocation("model", 3500, is_gpu=True)

# After inference
memory.print_stats()
```

### Bottleneck Identification

```python
# Profile each stage
profiler.start("load_model")
# ... load ...
profiler.end("load_model")

profiler.start("preprocess")
# ... preprocess ...
profiler.end("preprocess")

profiler.start("inference")
# ... inference ...
profiler.end("inference")

profiler.start("postprocess")
# ... postprocess ...
profiler.end("postprocess")

profiler.print_summary()  # See where time is spent
```

## Best Practices

### 1. Start Conservative

```python
# Begin with safe settings
config = Config(
    max_vram_usage_mb=6656,  # Leave headroom
    default_batch_size=1,
    use_quantization=False
)
```

### 2. Measure Everything

```python
# Always profile before optimizing
profiler = Profiler()
memory = MemoryManager()

# Baseline measurements
baseline_time = run_inference(config)
baseline_memory = memory.get_stats()

# After optimization
optimized_time = run_inference(optimized_config)
optimized_memory = memory.get_stats()

# Compare
speedup = baseline_time / optimized_time
memory_saving = baseline_memory.used_vram_gb - optimized_memory.used_vram_gb
```

### 3. Incremental Optimization

```python
# Try one optimization at a time
optimizations = [
    {'use_quantization': True},
    {'enable_cpu_offload': True},
    {'num_inference_steps': 15},
]

for opt in optimizations:
    config.update(opt)
    result = test_config(config)
    if result.quality >= threshold:
        # Keep optimization
        best_config.update(opt)
```

### 4. Document Results

```markdown
## Optimization Results

### Configuration
- Model: SD 2.1 Base
- Resolution: 512x512
- Batch size: 1

### Results
- Baseline: 45s, 7.2GB VRAM
- +Quantization (8-bit): 32s, 3.6GB VRAM ✅
- +Step reduction (50->15): 18s, 3.6GB VRAM ✅

### Final Configuration
- Quantization: 8-bit
- Steps: 15
- Total speedup: 2.5x
- Memory saving: 3.6GB
```

## Common Pitfalls

### 1. Over-optimization

```python
# Don't sacrifice quality for marginal gains
# Bad: 4-bit quantization + 5 steps = poor quality
# Good: 8-bit quantization + 15 steps = good balance
```

### 2. Ignoring Overhead

```python
# Consider full pipeline, not just inference
total_time = (
    model_load_time +  # Often neglected
    preprocessing_time +
    inference_time +
    postprocessing_time
)
```

### 3. Premature Optimization

```python
# Profile first, optimize second
# Don't assume bottlenecks
```

## Hardware-Specific Tips

### RX 580 8GB

```python
# Optimal settings for 8GB variant
config = Config(
    max_vram_usage_mb=7168,
    max_batch_size=2,  # For 512x512
)
```

### RX 580 4GB

```python
# Aggressive optimization for 4GB variant
config = Config(
    max_vram_usage_mb=3584,
    use_quantization=True,
    quantization_bits=8,
    enable_cpu_offload=True,
    max_batch_size=1,
)
```

## Performance Targets

### Stable Diffusion 2.1 (512x512)

| Configuration | Time | VRAM | Quality |
|--------------|------|------|---------|
| Baseline (FP32, 50 steps) | 60s | 7.5GB | 10/10 |
| Quantized (8-bit, 50 steps) | 42s | 4.0GB | 9.5/10 |
| Optimized (8-bit, 20 steps) | 18s | 4.0GB | 9/10 |
| Aggressive (8-bit, 15 steps) | 14s | 4.0GB | 8.5/10 |

### Goals

- **Target**: <20s for 512x512 image
- **Quality**: >8.5/10 user rating
- **Memory**: <4GB VRAM

---

For more optimization techniques, see:
- [Architecture Documentation](architecture.md)
- [Profiling Guide](profiling.md) (coming soon)
- [Benchmarks](../benchmarks/) (coming soon)
