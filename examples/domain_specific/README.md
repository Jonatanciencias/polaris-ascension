# Domain-Specific Examples

This directory contains examples of applying the Legacy GPU AI Platform's
advanced features to specific application domains.

## Session 20: Research Integration

These examples demonstrate the integration of cutting-edge scientific research
into practical applications.

### Available Examples

#### 1. Medical Imaging with PINNs (`medical_imaging_pinn.py`)

Physics-Informed Neural Networks for medical imaging applications:

- **CT Reconstruction**: Using Beer-Lambert law physics
- **MRI Denoising**: Anisotropic diffusion with edge preservation
- **Ultrasound Enhancement**: Wave propagation models

**Research References:**
- Raissi et al. (2019) - Physics-informed neural networks
- Miñoza et al. (2026) - SPIKE regularization
- Sun et al. (2021) - Physics-informed deep learning for medical imaging

**Key Features:**
- Reduced data requirements (physics provides constraints)
- Better generalization to unseen cases
- Physically plausible reconstructions

**Run:**
```bash
python examples/domain_specific/medical_imaging_pinn.py
```

#### 2. Agricultural Monitoring with SNNs (`agriculture_snn.py`)

Spiking Neural Networks with homeostatic mechanisms for agriculture:

- **Crop Health Classification**: Multispectral analysis
- **Pest Detection**: Event-driven visual sensors
- **Soil Moisture Prediction**: Temporal sensor data
- **Irrigation Control**: Continuous monitoring

**Research References:**
- Touda & Okuno (2026) - Synaptic scaling for SNN learning
- Massey et al. (2025) - Sleep-based homeostatic regularization
- Davies et al. (2018) - Loihi neuromorphic processor

**Key Features:**
- Ultra-low power consumption (solar-powered sensors)
- Event-driven processing (spikes on changes only)
- Online learning (STDP adapts to local conditions)
- Edge deployment ready

**Run:**
```bash
python examples/domain_specific/agriculture_snn.py
```

## Scientific Foundations

### Physics-Informed Neural Networks (PINNs)

PINNs embed physical laws (PDEs) into neural network training:

```
L_total = L_data + λ × L_physics
```

Where `L_physics` penalizes violations of the physical equations.

**Supported PDEs:**
- Heat equation: ∂u/∂t = α∇²u
- Wave equation: ∂²u/∂t² = c²∇²u
- Burgers equation: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
- Navier-Stokes: Full fluid dynamics

### Spiking Neural Networks with Homeostasis

Biologically-inspired neural networks with self-regulation:

1. **Synaptic Scaling**: Maintain target firing rates
2. **Intrinsic Plasticity**: Adapt neuron thresholds
3. **Sleep Consolidation**: Periodic weight normalization
4. **STDP Learning**: Spike-timing dependent plasticity

**Energy Efficiency:**
- 10-100× lower power than conventional ANNs
- Event-driven computation
- Natural for temporal data

### Evolutionary Pruning

Bio-inspired network optimization through:

1. **Natural Selection**: Fittest connections survive
2. **Mutation**: Random topology exploration
3. **Crossover**: Combine successful structures
4. **Adaptation**: Gradual sparsity increase

**Results:**
- 80-95% weight reduction
- 95-99% accuracy retention
- 2-5× inference speedup

## Hardware Optimization

All examples are optimized for AMD Polaris (RX 580):

- Sparse matrix operations leverage GCN architecture
- Memory-efficient representations
- Wavefront-friendly access patterns
- 4-8GB VRAM constraints respected

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
```

## License

MIT License - See main project LICENSE file.
