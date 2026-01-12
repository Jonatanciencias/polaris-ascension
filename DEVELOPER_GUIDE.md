# Developer Guide - Radeon RX 580 AI Framework

**Version**: 0.2.0  
**Status**: Production Ready (Core Framework)  
**Last Updated**: January 12, 2026

---

## 🎯 Quick Start for Developers

### Prerequisites

```bash
# System requirements
- AMD Radeon RX 580 (8GB VRAM recommended)
- Ubuntu 20.04+ / Debian-based Linux
- Python 3.8-3.12
- 16GB+ RAM
- OpenCL support (Mesa or ROCm)
```

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/radeon-rx580-ai.git
cd radeon-rx580-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python scripts/verify_hardware.py
pytest tests/ -v
```

---

## 📚 Framework Architecture

### Core Components

#### 1. GPU Management (`src/core/gpu.py`)
```python
from src.core.gpu import GPUManager

# Initialize GPU
gpu = GPUManager()
gpu.initialize()

# Get GPU info
info = gpu.get_info()
print(f"GPU: {info.name}")
print(f"VRAM: {info.vram_mb}MB")
print(f"OpenCL: {info.opencl_available}")
```

**Use Cases:**
- Hardware detection and validation
- Backend selection (OpenCL/ROCm/CPU)
- Driver compatibility checks

#### 2. Memory Management (`src/core/memory.py`)
```python
from src.core.memory import MemoryManager

# Track memory usage
memory = MemoryManager(gpu_vram_mb=8192)

# Register allocations
memory.register_allocation("model", 3500, is_gpu=True)
memory.register_allocation("batch", 512, is_gpu=True)

# Get recommendations
recs = memory.get_recommendations(model_size_mb=3500)
print(f"Max batch size: {recs['max_batch_size']}")
print(f"Strategy: {recs['strategy']}")

# Monitor usage
memory.print_stats()
```

**Use Cases:**
- VRAM allocation planning
- Batch size optimization
- Out-of-memory prevention

#### 3. Performance Profiling (`src/core/profiler.py`)
```python
from src.core.profiler import Profiler

# Profile operations
profiler = Profiler()

profiler.start("preprocessing")
# ... your code ...
profiler.end("preprocessing")

profiler.start("inference")
# ... your code ...
profiler.end("inference")

# Get statistics
summary = profiler.get_summary()
print(f"Inference avg: {summary['inference']['avg_ms']:.2f}ms")

# Print formatted table
profiler.print_summary()
```

**Use Cases:**
- Bottleneck identification
- Performance optimization
- Regression testing

### Inference System

#### 4. ONNX Engine (`src/inference/onnx_engine.py`)
```python
from src.inference import ONNXInferenceEngine, InferenceConfig

# Configure engine
config = InferenceConfig(
    device='auto',  # 'auto', 'opencl', 'cpu'
    precision='fp32',  # 'fp32', 'fp16', 'int8'
    batch_size=1,
    optimization_level=2  # 0=none, 1=basic, 2=aggressive
)

# Initialize
engine = ONNXInferenceEngine(config)

# Load model
engine.load_model('path/to/model.onnx')

# Run inference
result = engine.infer('image.jpg')
print(f"Top prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")

# Performance stats
engine.profiler.print_summary()
```

**Supported Inputs:**
- File path (str or Path)
- PIL Image
- NumPy array (preprocessed)

**Output Format:**
```python
{
    'predictions': [
        {'class_id': 281, 'class_name': 'tabby cat', 'confidence': 0.452},
        {'class_id': 282, 'class_name': 'tiger cat', 'confidence': 0.367},
        # ... top-5 ...
    ],
    'raw_output': np.ndarray,  # Raw logits
    'performance': {
        'preprocessing_ms': 15.2,
        'inference_ms': 6.1,
        'postprocessing_ms': 0.3,
        'total_ms': 21.6
    }
}
```

---

## 🔬 Mathematical Experiments Framework

### Precision Experiments (`src/experiments/precision_experiments.py`)

Test different precision levels for your application:

```python
from src.experiments.precision_experiments import PrecisionExperiment
import numpy as np

# Create experiment
experiment = PrecisionExperiment()

# Simulate medical imaging scenario
image_data = np.random.randn(3, 224, 224).astype(np.float32)
results = experiment.test_medical_imaging_precision(image_data)

# Check safety
print(f"FP16 SNR: {results['fp16']['snr_db']:.2f} dB")
print(f"Safe for diagnosis: {results['fp16']['diagnostic_quality']}")
print(f"Recommendation: {results['fp16']['recommendation']}")
```

**When to use:**
- Evaluating FP16/INT8 safety for your application
- Understanding precision-performance tradeoffs
- Validating medical/scientific accuracy requirements

### Sparse Networks (`src/experiments/sparse_networks.py`)

Implement sparse networks for memory-constrained scenarios:

```python
from src.experiments.sparse_networks import SparseNetwork, sparse_vs_dense_benchmark
import numpy as np

# Create sparse network
sparse_net = SparseNetwork(
    pruning_method='magnitude',  # 'magnitude', 'random', 'structured'
    sparsity=0.9  # 90% of weights are zero
)

# Prune dense weights
dense_weights = np.random.randn(2048, 2048).astype(np.float32)
sparse_weights = sparse_net.create_sparse_weights(dense_weights, sparsity=0.9)

# Run inference with sparse weights
input_data = np.random.randn(2048, 50).astype(np.float32)
output = sparse_net.sparse_matmul(sparse_weights, input_data)

# Benchmark sparse vs dense
results = sparse_vs_dense_benchmark(
    model_size=(2048, 2048),
    sparsity_levels=[0.0, 0.5, 0.7, 0.9, 0.95],
    num_iterations=100
)
```

**Benefits:**
- 10x memory reduction at 90% sparsity
- 5-8x speedup (depending on implementation)
- Enables larger models on limited VRAM

**Limitations:**
- Requires model retraining or fine-tuning
- Not all models benefit equally
- CPU sparse operations may be slower than dense GPU

### Quantization Analysis (`src/experiments/quantization_analysis.py`)

Validate quantization safety for critical applications:

```python
from src.experiments.quantization_analysis import QuantizationAnalyzer
import numpy as np

analyzer = QuantizationAnalyzer()

# Medical diagnosis safety
predictions = np.random.rand(1000, 10).astype(np.float32)  # Class probabilities
result = analyzer.test_medical_safety(
    predictions=predictions,
    bits=8,
    task='classification'
)

print(f"Decision stability: {result['decision_stability']:.4f}")
print(f"Is safe: {result['is_medically_safe']}")

# Genomic ranking preservation
scores = np.random.randn(10000).astype(np.float32)  # Variant scores
result = analyzer.test_genomic_ranking_preservation(
    scores=scores,
    bits=8,
    top_k=1000
)

print(f"Spearman correlation: {result['spearman_correlation']:.6f}")
print(f"Top-1000 overlap: {result['top_k_overlap']:.4f}")
```

---

## 🎓 Version-Specific Recommendations

### Version 0.2.0 (Current - Production Ready)

**What's Ready:**
- ✅ Core inference framework
- ✅ ONNX model support
- ✅ Mathematical optimization experiments
- ✅ Comprehensive benchmarking
- ✅ Production examples

**Recommended Use Cases:**
1. **Image Classification** (Production Ready)
   - Medical imaging screening
   - Wildlife conservation
   - Manufacturing QA
   - Agricultural disease detection

2. **Mathematical Research** (Experimentally Validated)
   - Precision requirement analysis
   - Sparse network feasibility studies
   - Quantization safety validation

**Not Recommended Yet:**
- ❌ Object detection (not implemented)
- ❌ Semantic segmentation (not implemented)
- ❌ Video processing (not optimized)
- ❌ Stable Diffusion (memory constraints)

### For Contributors

#### Adding New Models

```python
# 1. Convert model to ONNX
import torch
import torch.onnx

model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "your_model.onnx")

# 2. Test with framework
from src.inference import ONNXInferenceEngine, InferenceConfig

config = InferenceConfig(device='auto')
engine = ONNXInferenceEngine(config)
engine.load_model('your_model.onnx')

# 3. Validate performance
result = engine.infer('test_image.jpg')
engine.profiler.print_summary()

# 4. Test precision requirements
from src.experiments.precision_experiments import PrecisionExperiment
experiment = PrecisionExperiment()
# ... validate FP16/INT8 safety ...
```

#### Custom Preprocessing

```python
from src.inference.onnx_engine import ONNXInferenceEngine
import numpy as np

class CustomONNXEngine(ONNXInferenceEngine):
    def preprocess(self, inputs):
        """Override preprocessing for your model"""
        # Your custom preprocessing
        if isinstance(inputs, str):
            img = Image.open(inputs)
        else:
            img = inputs
            
        # Custom normalization
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.custom_mean) / self.custom_std
        
        # Add batch dimension
        return np.expand_dims(img, axis=0)
    
    def postprocess(self, outputs):
        """Override postprocessing for your model"""
        # Your custom postprocessing
        return {
            'predictions': self._extract_predictions(outputs),
            'raw_output': outputs
        }
```

#### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gpu.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
python scripts/benchmark.py --all
```

#### Code Style

```bash
# Format code
black src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Type checking (optional)
mypy src/
```

---

## 🚀 Performance Optimization Guide

### 1. Memory Optimization

```python
from src.core.memory import MemoryManager

memory = MemoryManager(gpu_vram_mb=8192)

# Check if allocation is feasible
if memory.can_allocate(size_mb=3500, use_gpu=True):
    # Load model
    engine.load_model('large_model.onnx')
    memory.register_allocation('model', 3500, is_gpu=True)

# Get optimization recommendations
recs = memory.get_recommendations(model_size_mb=3500)
optimal_batch = recs['max_batch_size']
```

**Strategies:**
- Use FP16: 2x memory reduction
- Use INT8: 4x memory reduction
- Use 90% sparsity: 10x memory reduction
- Combine: 20x memory reduction

### 2. Precision Optimization

```python
# Test precision requirements first
from src.experiments.precision_experiments import PrecisionExperiment

experiment = PrecisionExperiment()
results = experiment.test_medical_imaging_precision(your_data)

if results['fp16']['diagnostic_quality']:
    # FP16 is safe, use it
    config = InferenceConfig(precision='fp16')
elif results['int8']['screening_quality']:
    # INT8 is safe for screening
    config = InferenceConfig(precision='int8')
else:
    # Stick with FP32
    config = InferenceConfig(precision='fp32')
```

**Expected Speedups:**
- FP16: 1.5-2.0x faster
- INT8: 2.0-4.0x faster
- Sparse 90%: 5-8x faster (with custom kernels)

### 3. Batch Processing

```python
from src.inference import ONNXInferenceEngine, InferenceConfig

# Configure for batch processing
config = InferenceConfig(batch_size=4)
engine = ONNXInferenceEngine(config)

# Process multiple images
images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
results = [engine.infer(img) for img in images]

# Optimize batch size
memory = MemoryManager()
recs = memory.get_recommendations(model_size_mb=14)
optimal_batch = recs['max_batch_size']
```

**Batch Size Guidelines:**
- MobileNetV2: Up to 32 images
- ResNet-50: Up to 8 images
- EfficientNet-B0: Up to 16 images

### 4. Profiling Workflow

```python
from src.core.profiler import Profiler

profiler = Profiler()

# Profile each stage
profiler.start('load_model')
engine.load_model('model.onnx')
profiler.end('load_model')

profiler.start('inference')
for img in images:
    profiler.start('single_inference')
    result = engine.infer(img)
    profiler.end('single_inference')
profiler.end('inference')

# Identify bottlenecks
summary = profiler.get_summary()
bottleneck = max(summary.items(), key=lambda x: x[1]['total_ms'])
print(f"Bottleneck: {bottleneck[0]} ({bottleneck[1]['total_ms']:.2f}ms)")
```

---

## 🔧 Common Issues & Solutions

### Issue 1: "OpenCL not available"

**Solution:**
```bash
# Install OpenCL
sudo apt install opencl-icd-dev opencl-headers clinfo mesa-opencl-icd

# Verify
clinfo --list

# If still not working, fall back to CPU
config = InferenceConfig(device='cpu')
```

### Issue 2: "Out of memory"

**Solution:**
```python
# Check memory before loading
memory = MemoryManager()
if not memory.can_allocate(model_size_mb, use_gpu=True):
    # Use CPU offloading or reduce batch size
    config = InferenceConfig(device='cpu')
    # or
    config = InferenceConfig(batch_size=1)
```

### Issue 3: "Slow inference"

**Solution:**
```python
# 1. Profile to identify bottleneck
profiler = Profiler()
# ... profile your code ...
profiler.print_summary()

# 2. Optimize based on bottleneck:
# - If preprocessing is slow: cache/batch images
# - If inference is slow: use FP16/INT8
# - If postprocessing is slow: optimize extraction

# 3. Use optimization comparison
python examples/optimizations_comparison.py
```

### Issue 4: "Model accuracy dropped"

**Solution:**
```python
# Test precision impact
from src.experiments.quantization_analysis import QuantizationAnalyzer

analyzer = QuantizationAnalyzer()
result = analyzer.test_medical_safety(
    predictions=your_predictions,
    bits=8,
    task='classification'
)

if not result['is_medically_safe']:
    # Increase precision
    config = InferenceConfig(precision='fp16')  # or 'fp32'
```

---

## 📖 Additional Resources

### Documentation
- [Architecture](docs/architecture.md) - System design
- [Optimization](docs/optimization.md) - Performance tuning
- [Use Cases](docs/use_cases.md) - Real-world applications
- [Mathematical Innovation](docs/mathematical_innovation.md) - Research foundation

### Examples
- [Image Classification](examples/image_classification.py) - Production inference
- [Mathematical Experiments](examples/mathematical_experiments.py) - Research demos
- [Optimization Comparison](examples/optimizations_comparison.py) - Benchmarking

### External Resources
- [ONNX Runtime](https://onnxruntime.ai/)
- [AMD ROCm](https://rocmdocs.amd.com/)
- [OpenCL Programming](https://www.khronos.org/opencl/)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/contributing.md) for guidelines.

**Priority Areas:**
1. Custom OpenCL kernels for sparse operations
2. Additional model support (detection, segmentation)
3. Hardware-specific optimizations
4. Real-world deployment case studies

---

## 📝 Version History

### v0.2.0 (2026-01-12) - Production Ready
- ✅ Complete inference framework
- ✅ Mathematical optimization experiments
- ✅ Comprehensive benchmarking
- ✅ Production examples

### v0.1.0 (2026-01-08) - Foundation
- ✅ Initial project structure
- ✅ Core modules (GPU, Memory, Profiler)
- ✅ Testing framework
- ✅ CI/CD pipeline

---

**Questions?** Open an issue on GitHub or check the documentation.
