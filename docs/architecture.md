# System Architecture

## Overview

The Radeon RX 580 AI Framework is designed to enable modern AI workloads on AMD Polaris-based GPUs through a modular, extensible architecture.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Applications                     │
│              (CLI, API, Python Scripts)                  │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│                  Inference Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Stable     │  │  Custom      │  │   Future     │  │
│  │  Diffusion   │  │  Models      │  │   Models     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│                    Core Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │     GPU      │  │    Memory    │  │   Profiler   │  │
│  │   Manager    │  │   Manager    │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│                 Backend Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   OpenCL     │  │    ROCm      │  │   CPU        │  │
│  │              │  │  (optional)  │  │   Fallback   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│                 Hardware Layer                           │
│            AMD Radeon RX 580 (Polaris 20)               │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. GPU Manager (`src/core/gpu.py`)

**Responsibilities:**
- GPU detection and identification
- Driver verification (AMDGPU, Mesa)
- Compute backend selection (OpenCL/ROCm/CPU)
- Device initialization and resource management

**Key Classes:**
- `GPUManager`: Main interface for GPU operations
- `GPUInfo`: Data container for GPU information

### 2. Memory Manager (`src/core/memory.py`)

**Responsibilities:**
- VRAM and RAM tracking
- Allocation planning and recommendations
- Memory optimization strategies
- Peak usage monitoring

**Key Classes:**
- `MemoryManager`: Manages memory allocation
- `MemoryStats`: Memory statistics container

**Optimization Strategies:**
- Quantization recommendations (8/4-bit)
- CPU offloading decisions
- Batch size calculations

### 3. Profiler (`src/core/profiler.py`)

**Responsibilities:**
- Performance measurement
- Operation timing
- Bottleneck identification
- Statistical analysis

**Key Classes:**
- `Profiler`: Performance tracking
- `ProfileEntry`: Individual measurement

### 4. Inference Layer (`src/inference/`)

**Responsibilities:**
- Model loading and initialization
- Inference execution
- Model-specific optimizations
- Pipeline management

**Planned Components:**
- `BaseInference`: Abstract base class
- `StableDiffusionInference`: SD implementation
- `Optimizer`: Model optimization utilities

## Data Flow

### Inference Pipeline

```
1. User Request
   ↓
2. Configuration Loading
   - Load model config
   - Apply optimizations
   ↓
3. GPU Initialization
   - Detect hardware
   - Select backend
   ↓
4. Memory Planning
   - Calculate requirements
   - Apply optimization strategy
   ↓
5. Model Loading
   - Load weights
   - Apply quantization (if enabled)
   - Allocate GPU memory
   ↓
6. Inference Execution
   - Transfer data
   - Execute compute
   - Profile performance
   ↓
7. Result Processing
   - Transfer results
   - Post-process
   ↓
8. Cleanup
   - Free memory
   - Update statistics
```

## Backend Selection Strategy

```python
if ROCm available and compatible:
    use ROCm  # Best performance
elif OpenCL available:
    use OpenCL  # Good compatibility
else:
    use CPU fallback  # Slowest but always works
```

## Memory Management Strategy

### VRAM Allocation

```
8GB Total VRAM (RX 580)
├── 512MB: System reserve
├── Model weights (variable)
├── Activations (batch-dependent)
└── Intermediate buffers
```

### Optimization Hierarchy

1. **Fits in VRAM**: Direct GPU execution
2. **Quantization helps**: Apply 8-bit quantization
3. **Still too large**: Apply 4-bit or CPU offload
4. **Very large models**: Hybrid CPU/GPU execution

## Configuration Management

### Configuration Hierarchy

```
1. Default config (configs/default.yaml)
   ↓
2. User config file (optional)
   ↓
3. Environment variables
   ↓
4. Command-line arguments
```

## Extension Points

The framework is designed to be extensible:

### Adding New Models

1. Extend `BaseInference`
2. Implement model-specific optimizations
3. Register in model registry

### Adding New Backends

1. Implement backend interface
2. Add detection logic to `GPUManager`
3. Update configuration schema

### Adding Optimizations

1. Implement in `Optimizer` class
2. Add configuration options
3. Update memory recommendations

## Testing Strategy

### Unit Tests
- Core component functionality
- Configuration management
- Memory calculations

### Integration Tests
- GPU detection
- Backend initialization
- End-to-end inference (coming soon)

### Performance Tests
- Memory usage benchmarks
- Inference speed tests
- Optimization effectiveness

## Future Architecture Plans

### Phase 2: Core Inference
- PyTorch-ROCm integration
- ONNX Runtime backend
- Stable Diffusion implementation

### Phase 3: Optimization
- Custom kernel library
- Advanced memory management
- Multi-GPU support (future)

### Phase 4: Production
- REST API server
- Web UI
- Container orchestration

## Design Principles

1. **Modularity**: Each component is independent and testable
2. **Extensibility**: Easy to add new models and backends
3. **Performance**: Optimize for limited VRAM
4. **Reliability**: Graceful degradation and error handling
5. **Usability**: Clear APIs and good documentation
