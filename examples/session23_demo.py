"""
Unified Optimization Pipeline Demo - Session 23
================================================

Demonstrates end-to-end optimization pipeline combining all techniques.

Examples:
---------
1. Quick Optimization (one-liner)
2. Custom Configuration
3. Multi-target Comparison
4. Physics-Aware Optimization (PINNs)
5. Full Pipeline Report

Author: Session 23 Implementation
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional

from src.pipelines.unified_optimization import (
    UnifiedOptimizationPipeline,
    OptimizationTarget,
    OptimizationConfig,
    PipelineStage,
    quick_optimize
)


# ============================================================================
# Test Models
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    """Conv net for image classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PINN(nn.Module):
    """Physics-Informed Neural Network."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)  # (x, t) input
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)  # u(x,t) output
        
    def forward(self, xt):
        x = torch.tanh(self.fc1(xt))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        u = self.fc4(x)
        return u


# ============================================================================
# Helper Functions
# ============================================================================

def create_dummy_data(n_samples=100, input_size=784, n_classes=10, reshape_for_conv=False):
    """Create dummy dataset."""
    if reshape_for_conv:
        # For ConvNet: (batch, 1, 28, 28)
        X = torch.randn(n_samples, 1, 28, 28)
    else:
        # For MLP: (batch, input_size)
        X = torch.randn(n_samples, input_size)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return loader


def simple_eval_fn(model, loader):
    """Simple evaluation function."""
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                # Convert to same dtype as model
                if next(model.parameters()).dtype == torch.float16:
                    inputs = inputs.half()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    except Exception:
        # If evaluation fails, return 0
        return 0.0
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def print_separator(title):
    """Print section separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_model_info(name, model):
    """Print model information."""
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"\n{name}:")
    print(f"  Parameters: {n_params:,}")
    print(f"  Size: {size_mb:.2f} MB")


# ============================================================================
# Demos
# ============================================================================

def demo_1_quick_optimize():
    """
    Demo 1: Quick Optimization (One-Liner)
    =======================================
    
    Simplest way to optimize a model with one line of code.
    """
    print_separator("DEMO 1: Quick Optimization")
    
    print("\nCreating simple MLP...")
    model = SimpleMLP()
    val_loader = create_dummy_data(100, 784, 10)
    
    print_model_info("Original Model", model)
    
    print("\n🚀 Running quick optimization (balanced)...")
    start = time.time()
    
    optimized, metrics = quick_optimize(
        model,
        target="balanced",
        val_loader=val_loader,
        eval_fn=simple_eval_fn
    )
    
    elapsed = time.time() - start
    
    print_model_info("Optimized Model", optimized)
    
    print(f"\n📊 Metrics:")
    print(f"  Compression: {metrics['compression_ratio']:.2f}x")
    print(f"  Speedup: {metrics['speedup']:.2f}x")
    print(f"  Memory reduction: {metrics['memory_reduction']:.1%}")
    print(f"  Accuracy drop: {metrics['accuracy_drop']:.4f}")
    print(f"  Success: {'✅' if metrics['success'] else '❌'}")
    print(f"  Time: {elapsed:.2f}s")


def demo_2_custom_config():
    """
    Demo 2: Custom Configuration
    =============================
    
    Fine-tune optimization with custom configuration.
    """
    print_separator("DEMO 2: Custom Configuration")
    
    print("\nCreating ConvNet...")
    model = ConvNet()
    val_loader = create_dummy_data(100, 28*28, 10, reshape_for_conv=True)
    
    # Custom config with strict constraints
    config = OptimizationConfig(
        target=OptimizationTarget.SPEED,
        max_accuracy_drop=0.03,  # Max 3% drop
        min_speedup=1.5,  # Minimum 1.5x speedup
        enabled_stages=[
            PipelineStage.PRUNING,
            PipelineStage.QUANTIZATION,
            PipelineStage.FINE_TUNING
        ],
        auto_tune=True
    )
    
    print("\nConfiguration:")
    print(f"  Target: {config.target.value}")
    print(f"  Max accuracy drop: {config.max_accuracy_drop:.1%}")
    print(f"  Min speedup: {config.min_speedup:.1f}x")
    print(f"  Stages: {[s.value for s in config.enabled_stages]}")
    
    pipeline = UnifiedOptimizationPipeline(config=config)
    
    print("\n🚀 Running custom pipeline...")
    result = pipeline.optimize(
        model,
        val_loader=val_loader,
        eval_fn=simple_eval_fn
    )
    
    print(f"\n📊 Results:")
    print(f"  Compression: {result.compression_ratio:.2f}x")
    print(f"  Speedup: {result.speedup:.2f}x")
    print(f"  Memory reduction: {result.memory_reduction:.1%}")
    print(f"  Total time: {result.total_time:.2f}s")
    print(f"  Stages completed: {len(result.stage_results)}")


def demo_3_multi_target():
    """
    Demo 3: Multi-Target Comparison
    ================================
    
    Compare optimization with different targets.
    """
    print_separator("DEMO 3: Multi-Target Comparison")
    
    print("\nCreating model for comparison...")
    base_model = SimpleMLP()
    val_loader = create_dummy_data(100, 784, 10)
    
    print_model_info("Base Model", base_model)
    
    targets = [
        ("Accuracy", OptimizationTarget.ACCURACY),
        ("Balanced", OptimizationTarget.BALANCED),
        ("Speed", OptimizationTarget.SPEED),
        ("Memory", OptimizationTarget.MEMORY),
        ("Extreme", OptimizationTarget.EXTREME)
    ]
    
    print("\n🚀 Running optimization with different targets...")
    print("\n{:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Target", "Compression", "Speedup", "Memory↓", "Acc Drop"
    ))
    print("-" * 70)
    
    for name, target in targets:
        # Clone model for each target
        model = SimpleMLP()
        
        pipeline = UnifiedOptimizationPipeline(target)
        result = pipeline.optimize(
            model,
            val_loader=val_loader,
            eval_fn=simple_eval_fn
        )
        
        print("{:<12} {:<12.2f}x {:<12.2f}x {:<12.1%} {:<12.4f}".format(
            name,
            result.compression_ratio,
            result.speedup,
            result.memory_reduction,
            result.accuracy_drop
        ))


def demo_4_physics_aware():
    """
    Demo 4: Physics-Aware Optimization (PINN)
    ==========================================
    
    Optimize Physics-Informed Neural Network with physics constraints.
    """
    print_separator("DEMO 4: Physics-Aware Optimization")
    
    print("\nCreating PINN (Physics-Informed Neural Network)...")
    model = PINN()
    
    # Create dummy physics data (x, t) -> u
    X = torch.rand(100, 2) * 2 - 1  # x, t in [-1, 1]
    y = torch.sin(torch.pi * X[:, 0]) * torch.exp(-X[:, 1])  # u(x,t)
    dataset = torch.utils.data.TensorDataset(X, y.unsqueeze(1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    def pinn_eval(model, loader):
        """PINN evaluation (MSE)."""
        model.eval()
        mse = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                mse += F.mse_loss(outputs, targets).item()
        return 1.0 / (1.0 + mse)  # Convert to accuracy-like metric
    
    print_model_info("Original PINN", model)
    
    config = OptimizationConfig(
        target=OptimizationTarget.BALANCED,
        max_accuracy_drop=0.05,  # Stricter for physics
        enabled_stages=[
            PipelineStage.PRUNING,
            PipelineStage.QUANTIZATION
        ]
    )
    
    pipeline = UnifiedOptimizationPipeline(config=config)
    
    print("\n🚀 Running physics-aware optimization...")
    result = pipeline.optimize(
        model,
        val_loader=loader,
        eval_fn=pinn_eval
    )
    
    print_model_info("Optimized PINN", result.optimized_model)
    
    print(f"\n📊 Results:")
    print(f"  Compression: {result.compression_ratio:.2f}x")
    print(f"  Physics accuracy preserved: {(1 - result.accuracy_drop/result.original_accuracy):.1%}")
    print(f"  Memory saved: {result.memory_reduction:.1%}")


def demo_5_full_report():
    """
    Demo 5: Full Pipeline Report
    =============================
    
    Generate comprehensive optimization report.
    """
    print_separator("DEMO 5: Full Pipeline Report")
    
    print("\nCreating model...")
    model = ConvNet()
    val_loader = create_dummy_data(100, 28*28, 10, reshape_for_conv=True)
    
    config = OptimizationConfig(
        target=OptimizationTarget.BALANCED,
        enabled_stages=[
            PipelineStage.PRUNING,
            PipelineStage.QUANTIZATION,
            PipelineStage.MIXED_PRECISION,
            PipelineStage.FINE_TUNING
        ]
    )
    
    pipeline = UnifiedOptimizationPipeline(config=config)
    
    print("\n🚀 Running full pipeline...")
    result = pipeline.optimize(
        model,
        val_loader=val_loader,
        eval_fn=simple_eval_fn
    )
    
    # Generate detailed report
    print("\n")
    report = pipeline.generate_report(result)
    print(report)
    
    # Save report
    report_path = "optimization_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n💾 Report saved to: {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  UNIFIED OPTIMIZATION PIPELINE - SESSION 23")
    print("  End-to-End Model Optimization Demo")
    print("=" * 70)
    
    demos = [
        ("Quick Optimization", demo_1_quick_optimize),
        ("Custom Configuration", demo_2_custom_config),
        ("Multi-Target Comparison", demo_3_multi_target),
        ("Physics-Aware (PINN)", demo_4_physics_aware),
        ("Full Report", demo_5_full_report)
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-" * 70)
    
    try:
        for name, demo_fn in demos:
            demo_fn()
            time.sleep(0.5)  # Brief pause between demos
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return
    
    print("\n" + "=" * 70)
    print("  ✅ All demos completed successfully!")
    print("=" * 70)
    
    # Summary
    print("\n📋 SUMMARY:")
    print("  • Quick optimize: One-line model optimization")
    print("  • Custom config: Fine-tuned optimization control")
    print("  • Multi-target: Compare different optimization goals")
    print("  • Physics-aware: Optimize PINNs with constraints")
    print("  • Full report: Comprehensive optimization analysis")
    print("\n💡 Use quick_optimize() for fast prototyping!")
    print("💡 Use UnifiedOptimizationPipeline for production!")


if __name__ == "__main__":
    main()
