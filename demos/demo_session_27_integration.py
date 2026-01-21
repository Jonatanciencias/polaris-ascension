"""
Demo: DARTS + Tensor Decomposition Integration (Session 27)

Demonstrates multi-objective neural architecture search with compression.

This demo shows:
1. Creating integrated search configuration
2. Running architecture search with DARTS
3. Applying tensor decomposition compression
4. Evaluating multiple objectives (accuracy, latency, memory)
5. Finding Pareto-optimal architectures
6. Selecting best architecture based on preferences

Hardware Target: AMD Radeon RX 580 (8GB VRAM, Polaris architecture)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compute.darts_decomposition import (
    create_integrated_search,
    DecompositionMethod,
    CompressionConfig,
    HardwareConstraints,
    ArchitectureMetrics,
    DARTSDecompositionIntegration
)
from compute.nas_darts import DARTSConfig


def create_cifar10_like_dataset(num_samples=500):
    """
    Create CIFAR-10 like synthetic dataset for demo.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        train_loader, val_loader
    """
    print(f"Creating synthetic dataset ({num_samples} samples)...")
    
    # Generate random images and labels
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    
    X_val = torch.randn(num_samples // 5, 3, 32, 32)
    y_val = torch.randint(0, 10, (num_samples // 5,))
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def print_architecture_metrics(name: str, metrics: ArchitectureMetrics):
    """Pretty print architecture metrics"""
    print(f"\n{name}:")
    print(f"  Accuracy:      {metrics.accuracy:.3f} ({metrics.accuracy*100:.1f}%)")
    print(f"  Latency:       {metrics.latency_ms:.2f} ms")
    print(f"  Memory:        {metrics.memory_mb:.1f} MB")
    print(f"  Parameters:    {metrics.params/1e6:.2f}M")
    print(f"  FLOPs:         {metrics.flops/1e9:.2f}G")
    print(f"  Compression:   {metrics.compression_ratio:.2f}x")
    print(f"  Power:         {metrics.power_estimate_watts:.1f}W")
    print(f"  Pareto Rank:   {metrics.pareto_rank}")


def demo_basic_integration():
    """Demo 1: Basic integration setup"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Integration Setup")
    print("="*80)
    
    # Create integration with factory function
    integration = create_integrated_search(
        num_classes=10,
        target_compression=2.0,
        max_memory_mb=4000.0  # Conservative for RX 580
    )
    
    print(f"\n✓ Created DARTS + Decomposition Integration")
    print(f"  Device: {integration.device}")
    print(f"  Target compression: {integration.compression_config.target_compression}x")
    print(f"  Max memory: {integration.hardware_constraints.max_memory_mb}MB")
    print(f"  Decomposition method: {integration.compression_config.method.value}")
    
    return integration


def demo_architecture_search(integration, train_loader, val_loader):
    """Demo 2: Run architecture search"""
    print("\n" + "="*80)
    print("DEMO 2: Architecture Search with Compression")
    print("="*80)
    
    print("\nSearching for optimal architectures...")
    print("  This will:")
    print("  1. Search for architectures with DARTS")
    print("  2. Apply tensor decomposition compression")
    print("  3. Evaluate on multiple objectives")
    print("  4. Compute Pareto frontier")
    
    start_time = time.time()
    
    # Run search with small settings for demo
    pareto_optimal = integration.search_and_compress(
        train_loader=train_loader,
        val_loader=val_loader,
        search_epochs=1,  # Minimal for demo
        finetune_epochs=1,
        num_candidates=3
    )
    
    search_time = time.time() - start_time
    
    print(f"\n✓ Search completed in {search_time:.1f}s")
    print(f"  Found {len(pareto_optimal)} Pareto-optimal architectures")
    
    return pareto_optimal


def demo_pareto_analysis(pareto_optimal, integration):
    """Demo 3: Analyze Pareto-optimal architectures"""
    print("\n" + "="*80)
    print("DEMO 3: Pareto-Optimal Architectures Analysis")
    print("="*80)
    
    print(f"\nFound {len(pareto_optimal)} Pareto-optimal architectures:")
    
    for i, (model, metrics) in enumerate(pareto_optimal):
        print_architecture_metrics(f"Architecture {i+1}", metrics)
    
    # Print comparison table
    print("\n" + "-"*80)
    print("Comparison Table:")
    print("-"*80)
    print(f"{'Arch':<6} {'Acc':<8} {'Latency':<10} {'Memory':<10} {'Params':<10} {'Compression':<12}")
    print("-"*80)
    
    for i, (model, metrics) in enumerate(pareto_optimal):
        print(f"#{i+1:<5} "
              f"{metrics.accuracy:.3f}    "
              f"{metrics.latency_ms:.2f}ms     "
              f"{metrics.memory_mb:.1f}MB     "
              f"{metrics.params/1e6:.2f}M     "
              f"{metrics.compression_ratio:.2f}x")
    
    print("-"*80)


def demo_preference_selection(pareto_optimal, integration):
    """Demo 4: Select architecture based on preferences"""
    print("\n" + "="*80)
    print("DEMO 4: Architecture Selection by Preference")
    print("="*80)
    
    preferences = ["balanced", "accuracy", "latency", "memory"]
    
    for preference in preferences:
        print(f"\n--- Preference: {preference.upper()} ---")
        
        best_model, best_metrics = integration.mo_optimizer.select_best_architecture(
            pareto_optimal,
            preference=preference
        )
        
        print(f"Selected architecture:")
        print(f"  Accuracy: {best_metrics.accuracy:.3f}")
        print(f"  Latency:  {best_metrics.latency_ms:.2f}ms")
        print(f"  Memory:   {best_metrics.memory_mb:.1f}MB")
        print(f"  Params:   {best_metrics.params/1e6:.2f}M")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 32, 32).to(integration.device)
        with torch.no_grad():
            output = best_model(dummy_input)
        print(f"  Output shape: {output.shape}")


def demo_compression_comparison(integration, train_loader, val_loader):
    """Demo 5: Compare different compression methods"""
    print("\n" + "="*80)
    print("DEMO 5: Compression Method Comparison")
    print("="*80)
    
    methods = [
        DecompositionMethod.TUCKER,
        DecompositionMethod.CP,
        DecompositionMethod.TENSOR_TRAIN
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.value.upper()} decomposition ---")
        
        # Update compression config
        integration.compression_config.method = method
        
        # Create a simple model
        model = integration._create_candidate_model(integration.darts_config)
        
        # Get original metrics
        original_params = sum(p.numel() for p in model.parameters())
        print(f"  Original params: {original_params/1e6:.2f}M")
        
        # Apply decomposition
        start = time.time()
        compressed = integration._decompose_model(model, method)
        decomp_time = time.time() - start
        
        compressed_params = sum(p.numel() for p in compressed.parameters())
        actual_compression = original_params / compressed_params if compressed_params > 0 else 1.0
        
        print(f"  Compressed params: {compressed_params/1e6:.2f}M")
        print(f"  Compression ratio: {actual_compression:.2f}x")
        print(f"  Decomposition time: {decomp_time:.3f}s")
        
        results[method.value] = {
            'compression': actual_compression,
            'time': decomp_time
        }
    
    # Summary
    print("\n" + "-"*80)
    print("Summary:")
    print("-"*80)
    print(f"{'Method':<15} {'Compression':<15} {'Time':<10}")
    print("-"*80)
    for method, data in results.items():
        print(f"{method:<15} {data['compression']:.2f}x           {data['time']:.3f}s")
    print("-"*80)


def demo_hardware_aware_search():
    """Demo 6: Hardware-aware search for RX 580"""
    print("\n" + "="*80)
    print("DEMO 6: Hardware-Aware Search for AMD RX 580")
    print("="*80)
    
    # RX 580 specifications
    print("\nAMD Radeon RX 580 Specifications:")
    print("  VRAM: 8GB GDDR5")
    print("  TDP: 185W")
    print("  Architecture: Polaris (GCN 4.0)")
    print("  Compute Units: 36")
    print("  Stream Processors: 2304")
    
    # Create hardware-specific constraints
    rx580_constraints = HardwareConstraints(
        max_memory_mb=6000.0,  # Leave margin for OS
        target_latency_ms=30.0,  # Real-time target
        max_power_watts=185.0,
        compute_capability="polaris"
    )
    
    compression_config = CompressionConfig(
        method=DecompositionMethod.AUTO,
        target_compression=3.0,  # More aggressive for edge deployment
        min_rank_ratio=0.1,
        max_rank_ratio=0.4
    )
    
    search_config = DARTSConfig(
        num_cells=6,
        num_nodes=4,
        init_channels=16,
        layers=14
    )
    # Add num_classes separately
    search_config.num_classes = 10
    
    from compute.darts_decomposition import DARTSDecompositionIntegration
    
    integration = DARTSDecompositionIntegration(
        darts_config=search_config,
        compression_config=compression_config,
        hardware_constraints=rx580_constraints,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\n✓ Created RX 580-optimized search")
    print(f"  Target latency: {rx580_constraints.target_latency_ms}ms")
    print(f"  Max memory: {rx580_constraints.max_memory_mb}MB")
    print(f"  Target compression: {compression_config.target_compression}x")


def demo_multi_objective_tradeoffs(pareto_optimal):
    """Demo 7: Visualize multi-objective trade-offs"""
    print("\n" + "="*80)
    print("DEMO 7: Multi-Objective Trade-offs")
    print("="*80)
    
    print("\nAccuracy vs Latency Trade-off:")
    print("-" * 60)
    print(f"{'Accuracy':<12} {'Latency (ms)':<15} {'Trade-off Score':<15}")
    print("-" * 60)
    
    for model, metrics in pareto_optimal:
        # Compute trade-off score (higher is better)
        # Balance accuracy (maximize) and latency (minimize)
        trade_off = metrics.accuracy / (metrics.latency_ms / 100.0)
        
        print(f"{metrics.accuracy:.3f}        "
              f"{metrics.latency_ms:.2f}           "
              f"{trade_off:.3f}")
    
    print("\nAccuracy vs Memory Trade-off:")
    print("-" * 60)
    print(f"{'Accuracy':<12} {'Memory (MB)':<15} {'Trade-off Score':<15}")
    print("-" * 60)
    
    for model, metrics in pareto_optimal:
        # Compute trade-off score
        trade_off = metrics.accuracy / (metrics.memory_mb / 1000.0)
        
        print(f"{metrics.accuracy:.3f}        "
              f"{metrics.memory_mb:.1f}           "
              f"{trade_off:.3f}")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("DARTS + TENSOR DECOMPOSITION INTEGRATION DEMO")
    print("Session 27: Multi-Objective Neural Architecture Search")
    print("="*80)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create dataset
    train_loader, val_loader = create_cifar10_like_dataset(num_samples=500)
    
    # Demo 1: Basic setup
    integration = demo_basic_integration()
    
    # Demo 2: Architecture search
    pareto_optimal = demo_architecture_search(integration, train_loader, val_loader)
    
    # Demo 3: Pareto analysis
    demo_pareto_analysis(pareto_optimal, integration)
    
    # Demo 4: Preference selection
    demo_preference_selection(pareto_optimal, integration)
    
    # Demo 5: Compression comparison
    demo_compression_comparison(integration, train_loader, val_loader)
    
    # Demo 6: Hardware-aware search
    demo_hardware_aware_search()
    
    # Demo 7: Multi-objective trade-offs
    demo_multi_objective_tradeoffs(pareto_optimal)
    
    # Final summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. DARTS + Decomposition enables multi-objective optimization")
    print("  2. Pareto frontier provides multiple optimal trade-offs")
    print("  3. Different preferences select different architectures")
    print("  4. Compression methods have different characteristics")
    print("  5. Hardware-aware search targets specific constraints")
    print("  6. Trade-off analysis helps understand design choices")
    
    print("\nNext Steps:")
    print("  - Train discovered architectures on real CIFAR-10 data")
    print("  - Deploy on AMD RX 580 and measure actual performance")
    print("  - Extend to larger datasets (CIFAR-100, ImageNet)")
    print("  - Integrate with production inference pipeline")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
