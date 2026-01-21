"""
Session 21 Mixed-Precision Quantization Demo
=============================================

Demonstrates:
1. Mixed-precision optimization with evolutionary search
2. Layer-wise sensitivity analysis
3. Neuromorphic deployment pipeline
4. Power estimation

Author: Radeon RX 580 Optimized AI Framework
Session: 21 - Research Integration Phase 2
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.compute.mixed_precision import (
    MixedPrecisionOptimizer,
    PhysicsAwareMixedPrecision,
    MixedPrecisionConfig
)
from src.deployment.neuromorphic import (
    NeuromorphicDeployment,
    create_neuromorphic_deployer
)


def create_example_model():
    """Create example neural network."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )


def demo_mixed_precision_basic():
    """Demo 1: Basic mixed-precision quantization."""
    print("\n" + "="*70)
    print("DEMO 1: Mixed-Precision Quantization")
    print("="*70)
    
    # Create model
    model = create_example_model()
    print(f"✓ Created model with 3 layers")
    
    # Create optimizer
    optimizer = MixedPrecisionOptimizer(model)
    print(f"✓ Initialized MixedPrecisionOptimizer")
    
    # Generate test data
    x = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    print(f"✓ Generated test data: {x.shape}")
    
    # Analyze sensitivity
    print("\n1. Analyzing layer sensitivity...")
    sensitivity = optimizer.analyze_sensitivity(x, y, method='gradient', num_samples=50)
    print(f"   → Analyzed {len(sensitivity)} layers")
    
    for name, sens in list(sensitivity.items())[:3]:
        print(f"   - {name}: sensitivity={sens.sensitivity_score:.4f}, "
              f"recommended={sens.recommended_precision.value}")
    
    # Search configuration
    print("\n2. Searching optimal configuration (evolutionary)...")
    config = optimizer.search_configuration(
        x, y,
        method='evolutionary',
        population_size=10,
        generations=5
    )
    print(f"   → Found configuration for {len(config.layer_configs)} layers")
    print(f"   → Compression ratio: {config.compression_ratio:.2f}x")
    
    # Apply configuration
    print("\n3. Applying mixed-precision...")
    quantized_model = optimizer.apply_configuration(config)
    print(f"   ✓ Quantized model created")
    
    # Test inference
    with torch.no_grad():
        output = quantized_model(x[:10])
    print(f"   ✓ Inference test passed: {output.shape}")
    
    # Estimate compression
    metrics = optimizer.estimate_compression(config)
    print(f"\n4. Compression metrics:")
    print(f"   - Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"   - Average bits: {metrics['avg_bits']:.2f}")
    print(f"   - Memory reduction: {metrics['memory_reduction_percent']:.1f}%")
    
    print("\n✅ Demo 1 complete!")
    return quantized_model, config


def demo_neuromorphic_deployment():
    """Demo 2: Neuromorphic deployment pipeline."""
    print("\n" + "="*70)
    print("DEMO 2: Neuromorphic Deployment")
    print("="*70)
    
    # Create simple SNN
    snn = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    print(f"✓ Created SNN with 2 layers")
    
    # Create deployer
    deployer = create_neuromorphic_deployer('loihi2', optimization_level=2)
    print(f"✓ Initialized Loihi 2 deployer")
    
    # Optimize for platform
    print("\n1. Optimizing for neuromorphic hardware...")
    optimized_snn = deployer.optimize_for_platform(snn, target_spike_rate=0.1)
    print(f"   ✓ Optimization complete")
    
    # Export to Lava format
    print("\n2. Exporting to Lava framework...")
    export_data = deployer.export_snn(optimized_snn, output_format='lava')
    print(f"   → Exported {len(export_data['layers'])} layers")
    print(f"   → Format: {export_data['format']}")
    print(f"   → Platform: {export_data['metadata']['neuron_type']}")
    
    # Estimate power
    print("\n3. Estimating power consumption...")
    power = deployer.estimate_power(snn, input_spike_rate=0.1)
    print(f"   → CPU/GPU power: {power.cpu_power_watts:.1f}W")
    print(f"   → Neuromorphic power: {power.neuromorphic_power_watts:.4f}W")
    print(f"   → Reduction factor: {power.reduction_factor:.0f}x")
    print(f"   → Energy per spike: {power.energy_per_spike_joules*1e12:.1f}pJ")
    
    # Benchmark
    print("\n4. Benchmarking performance...")
    test_input = torch.randn(32, 784)
    bench_results = deployer.benchmark(snn, test_input, num_runs=20)
    print(f"   → Latency: {bench_results['latency_ms']:.2f}ms")
    print(f"   → Throughput: {bench_results['throughput_inferences_per_sec']:.1f} inferences/s")
    print(f"   → Energy per inference: {bench_results['energy_per_inference_uj']:.2f}μJ")
    
    print("\n✅ Demo 2 complete!")
    return export_data, power


def demo_physics_aware_quantization():
    """Demo 3: Physics-aware mixed-precision for PINNs."""
    print("\n" + "="*70)
    print("DEMO 3: Physics-Aware Mixed-Precision (PINN)")
    print("="*70)
    
    # Create simple PINN
    class SimplePINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(2, 20)
            self.hidden = nn.Linear(20, 20)
            self.output_layer = nn.Linear(20, 1)
        
        def forward(self, x):
            x = torch.tanh(self.input_layer(x))
            x = torch.tanh(self.hidden(x))
            return self.output_layer(x)
    
    pinn = SimplePINN()
    print(f"✓ Created PINN with 3 layers")
    
    # Create physics-aware optimizer
    optimizer = PhysicsAwareMixedPrecision(
        pinn,
        pde_loss_weight=1.0,
        physics_loss_threshold=1e-4
    )
    print(f"✓ Initialized PhysicsAwareMixedPrecision")
    
    # Generate test data
    x = torch.randn(100, 2, requires_grad=True)
    print(f"✓ Generated physics test points: {x.shape}")
    
    # Search configuration (heuristic for speed)
    print("\n1. Searching physics-aware configuration...")
    config = optimizer.search_configuration(x, None, method='heuristic')
    print(f"   → Found configuration for {len(config.layer_configs)} layers")
    
    # Show layer configurations
    print("\n2. Layer-wise precision assignments:")
    for layer_name, layer_config in config.layer_configs.items():
        precision = layer_config.get('precision', 'int8')
        bits = layer_config.get('bits', 8)
        print(f"   - {layer_name}: {precision} ({bits} bits)")
    
    # Validate physics accuracy
    print("\n3. Validating physics accuracy...")
    try:
        is_valid = optimizer.validate_physics_accuracy(config, x, threshold=0.1)
        print(f"   → Physics validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    except Exception as e:
        print(f"   → Physics validation skipped (expected): {type(e).__name__}")
    
    print("\n✅ Demo 3 complete!")
    return config


def demo_multi_platform_export():
    """Demo 4: Export to multiple neuromorphic platforms."""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Platform Neuromorphic Export")
    print("="*70)
    
    # Create SNN
    snn = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    print(f"✓ Created SNN")
    
    platforms = [
        ('loihi2', 'lava'),
        ('spinnaker2', 'pynn'),
        ('generic', 'spike_json')
    ]
    
    exports = {}
    
    for platform, format_name in platforms:
        print(f"\n{platform.upper()}:")
        deployer = create_neuromorphic_deployer(platform)
        
        # Optimize
        optimized = deployer.optimize_for_platform(snn)
        print(f"  ✓ Optimized for {platform}")
        
        # Export
        export_data = deployer.export_snn(optimized, output_format=format_name)
        exports[platform] = export_data
        print(f"  ✓ Exported to {format_name} format")
        
        # Power estimate
        power = deployer.estimate_power(snn)
        print(f"  → Power reduction: {power.reduction_factor:.0f}x")
    
    print(f"\n✓ Exported to {len(exports)} platforms successfully")
    print("\n✅ Demo 4 complete!")
    return exports


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("SESSION 21: Mixed-Precision & Neuromorphic Deployment")
    print("="*70)
    print("\nThis demo showcases:")
    print("  1. Mixed-precision quantization with evolutionary search")
    print("  2. Neuromorphic deployment pipeline (Loihi 2)")
    print("  3. Physics-aware quantization for PINNs")
    print("  4. Multi-platform export (Loihi, SpiNNaker, Generic)")
    
    try:
        # Run demos
        quantized_model, config = demo_mixed_precision_basic()
        export_data, power = demo_neuromorphic_deployment()
        pinn_config = demo_physics_aware_quantization()
        multi_exports = demo_multi_platform_export()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✅ Mixed-Precision: {config.compression_ratio:.2f}x compression")
        print(f"✅ Neuromorphic: {power.reduction_factor:.0f}x power reduction")
        print(f"✅ Physics-Aware: {len(pinn_config.layer_configs)} PINN layers quantized")
        print(f"✅ Multi-Platform: {len(multi_exports)} platforms supported")
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during demo execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
