"""
Session 22 Demo - PINN Interpretability + GNN Optimization
===========================================================

Demonstrates the key features implemented in Session 22:
1. PINN Interpretability - Sensitivity maps and residual analysis
2. GNN Optimization - ROCm-optimized graph neural networks

Author: Session 22
Date: January 2026
"""

import time

import numpy as np
import torch
import torch.nn as nn

from src.compute.gnn_optimization import (
    GATConv,
    GraphBatch,
    GraphSAGEConv,
    OptimizedGCN,
    create_karate_club_graph,
)

# Import Session 22 modules
from src.compute.pinn_interpretability import (
    PINNInterpreter,
    burgers_equation_residual,
    heat_equation_residual,
    wave_equation_residual,
)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    """Print formatted subheader."""
    print(f"\n{title}")
    print("-" * 50)


# ============================================================================
# DEMO 1: PINN Interpretability - Sensitivity Analysis
# ============================================================================


def demo1_pinn_sensitivity():
    """Demo PINN sensitivity analysis."""
    print_header("DEMO 1: PINN Interpretability - Sensitivity Analysis")

    # Create simple PINN
    print_subheader("1. Creating PINN model")

    class SimplePINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    pinn = SimplePINN()
    print(f"✓ Created PINN with {sum(p.numel() for p in pinn.parameters())} parameters")

    # Create interpreter
    print_subheader("2. Initializing interpreter")
    interpreter = PINNInterpreter(pinn, input_names=["x", "t"])
    print("✓ Interpreter initialized")

    # Test points
    print_subheader("3. Generating test points")
    x = torch.linspace(0, 1, 20)
    t = torch.linspace(0, 1, 20)
    X, T = torch.meshgrid(x, t, indexing="ij")
    test_points = torch.stack([X.flatten(), T.flatten()], dim=1)
    print(f"✓ Generated {len(test_points)} test points")

    # Compute sensitivity maps
    print_subheader("4. Computing sensitivity maps")
    start = time.time()

    # Gradient-based sensitivity
    grad_result = interpreter.compute_sensitivity_map(test_points, method="gradient")

    # Integrated gradients
    ig_result = interpreter.compute_sensitivity_map(test_points, method="integrated_gradients")

    # SmoothGrad
    smooth_result = interpreter.compute_sensitivity_map(test_points, method="smooth_grad")

    elapsed = time.time() - start

    print(f"✓ Computed 3 sensitivity methods in {elapsed:.3f}s")
    print(
        f"  → Gradient sensitivity: x={grad_result.feature_importance['x']:.3f}, t={grad_result.feature_importance['t']:.3f}"
    )
    print(
        f"  → Integrated gradients: x={ig_result.feature_importance['x']:.3f}, t={ig_result.feature_importance['t']:.3f}"
    )
    print(
        f"  → SmoothGrad: x={smooth_result.feature_importance['x']:.3f}, t={smooth_result.feature_importance['t']:.3f}"
    )

    # Feature importance
    print_subheader("5. Feature importance analysis")
    importance = interpreter.feature_importance(test_points)
    print(f"✓ Feature importance:")
    for feature, score in importance.items():
        print(f"  → {feature}: {score:.3f} ({score*100:.1f}%)")

    print("\n✅ Demo 1 complete!")
    return {
        "sensitivity_methods": 3,
        "test_points": len(test_points),
        "feature_importance": importance,
    }


# ============================================================================
# DEMO 2: PINN Interpretability - Physics Residual Analysis
# ============================================================================


def demo2_pinn_residual():
    """Demo PINN physics residual analysis."""
    print_header("DEMO 2: PINN Interpretability - Physics Residual Analysis")

    # Create PINN
    print_subheader("1. Creating PINN model")

    class HeatPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)

    pinn = HeatPINN()
    print(f"✓ Created heat equation PINN")

    # Create interpreter
    interpreter = PINNInterpreter(pinn, input_names=["x", "t"])

    # Domain points
    print_subheader("2. Generating domain points")
    x = torch.linspace(0, 1, 15)
    t = torch.linspace(0, 1, 15)
    X, T = torch.meshgrid(x, t, indexing="ij")
    domain_points = torch.stack([X.flatten(), T.flatten()], dim=1)
    print(f"✓ Generated {len(domain_points)} domain points (15x15 grid)")

    # Analyze residual
    print_subheader("3. Analyzing physics residual")
    start = time.time()

    # Heat equation: ∂u/∂t - α * ∂²u/∂x² = 0
    analysis = interpreter.analyze_residual(
        domain_points,
        lambda inputs, u: heat_equation_residual(inputs, u, alpha=0.01),
        grid_shape=(15, 15),
    )

    elapsed = time.time() - start

    print(f"✓ Analyzed residual in {elapsed:.3f}s")
    print(f"\nResidual statistics:")
    for stat_name, stat_value in analysis.residual_stats.items():
        print(f"  → {stat_name}: {stat_value:.6f}")

    # Hotspots
    print_subheader("4. Identifying high-error hotspots")
    print(f"✓ Found {len(analysis.hotspots)} hotspots:")
    for i, (x_val, t_val) in enumerate(analysis.hotspots[:5], 1):
        print(f"  → Hotspot {i}: x={x_val:.3f}, t={t_val:.3f}")

    print("\n✅ Demo 2 complete!")
    return {
        "domain_points": len(domain_points),
        "mean_residual": analysis.residual_stats["mean"],
        "num_hotspots": len(analysis.hotspots),
    }


# ============================================================================
# DEMO 3: PINN Interpretability - Layer Activation Analysis
# ============================================================================


def demo3_pinn_activations():
    """Demo PINN layer activation analysis."""
    print_header("DEMO 3: PINN Interpretability - Layer Activation Analysis")

    # Create PINN
    print_subheader("1. Creating PINN model")

    class DeepPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    pinn = DeepPINN()
    print(f"✓ Created deep PINN with 4 layers")

    # Create interpreter
    interpreter = PINNInterpreter(pinn, input_names=["x", "t"])

    # Test points
    test_points = torch.randn(50, 2)

    # Analyze activations
    print_subheader("2. Analyzing layer activations")
    start = time.time()
    activations = interpreter.analyze_layer_activations(test_points)
    elapsed = time.time() - start

    print(f"✓ Analyzed {len(activations)} layers in {elapsed:.3f}s")

    print_subheader("3. Activation statistics per layer")
    for i, (layer_name, stats) in enumerate(activations.items(), 1):
        print(f"\nLayer {i}: {layer_name}")
        print(f"  → Mean: {stats['mean']:.6f}")
        print(f"  → Std: {stats['std']:.6f}")
        print(f"  → Sparsity: {stats['sparsity']:.3f} ({stats['sparsity']*100:.1f}%)")
        print(f"  → Dead neurons: {stats['dead_neurons']}")

    # Gradient statistics
    print_subheader("4. Gradient flow analysis")
    grad_stats = interpreter.gradient_statistics(test_points)

    print(f"✓ Computed gradients for {len(grad_stats)} parameter tensors")
    total_norm = sum(stats["norm"] for stats in grad_stats.values())
    print(f"  → Total gradient norm: {total_norm:.6f}")

    print("\n✅ Demo 3 complete!")
    return {"num_layers": len(activations), "total_gradient_norm": total_norm}


# ============================================================================
# DEMO 4: GNN Optimization - Graph Convolutional Network
# ============================================================================


def demo4_gnn_gcn():
    """Demo optimized GCN."""
    print_header("DEMO 4: GNN Optimization - Graph Convolutional Network")

    # Create graph
    print_subheader("1. Loading Karate Club graph")
    graph = create_karate_club_graph()
    print(f"✓ Loaded graph:")
    print(f"  → Nodes: {graph.x.size(0)}")
    print(f"  → Edges: {graph.edge_index.size(1)}")

    # Create optimized GCN
    print_subheader("2. Creating optimized GCN")
    model = OptimizedGCN(
        in_channels=34,
        hidden_channels=64,
        num_layers=3,
        out_channels=16,
        dropout=0.5,
        optimization_level=2,  # Aggressive optimization
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created 3-layer GCN:")
    print(f"  → Parameters: {num_params:,}")
    print(f"  → Architecture: 34 → 64 → 64 → 16")
    print(f"  → Optimization level: 2 (aggressive)")

    # Forward pass
    print_subheader("3. Testing forward pass")
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)

    print(f"✓ Forward pass successful:")
    print(f"  → Input shape: {graph.x.shape}")
    print(f"  → Output shape: {out.shape}")

    # Benchmark
    print_subheader("4. Benchmarking performance")
    result = model.benchmark(graph, num_iterations=100)

    print(f"✓ Benchmark results (100 iterations):")
    print(f"  → Throughput: {result.throughput:.1f} graphs/s")
    print(f"  → Latency: {result.latency:.3f} ms/graph")
    print(f"  → Memory used: {result.memory_used:.2f} MB")

    print("\n✅ Demo 4 complete!")
    return {
        "num_nodes": graph.x.size(0),
        "num_edges": graph.edge_index.size(1),
        "throughput": result.throughput,
        "latency": result.latency,
    }


# ============================================================================
# DEMO 5: GNN Optimization - Multi-Architecture Comparison
# ============================================================================


def demo5_gnn_comparison():
    """Demo comparison of GNN architectures."""
    print_header("DEMO 5: GNN Optimization - Architecture Comparison")

    # Create test graph
    print_subheader("1. Creating test graph")
    num_nodes = 100
    num_edges = 500
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    graph = GraphBatch(x=x, edge_index=edge_index, num_graphs=1)

    print(f"✓ Created random graph:")
    print(f"  → Nodes: {num_nodes}")
    print(f"  → Edges: {num_edges}")

    # Test different architectures
    architectures = {
        "GCN": OptimizedGCN(32, 64, num_layers=2, out_channels=16),
        "GCN-Deep": OptimizedGCN(32, 64, num_layers=4, out_channels=16),
        "GCN-Wide": OptimizedGCN(32, 128, num_layers=2, out_channels=16),
    }

    print_subheader("2. Comparing architectures")
    results = {}

    for name, model in architectures.items():
        model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Benchmark
        benchmark = model.benchmark(graph, num_iterations=50)

        results[name] = {
            "params": num_params,
            "throughput": benchmark.throughput,
            "latency": benchmark.latency,
        }

        print(f"\n{name}:")
        print(f"  → Parameters: {num_params:,}")
        print(f"  → Throughput: {benchmark.throughput:.1f} graphs/s")
        print(f"  → Latency: {benchmark.latency:.3f} ms/graph")

    # Find best
    print_subheader("3. Performance comparison")
    best_throughput = max(results.items(), key=lambda x: x[1]["throughput"])
    best_latency = min(results.items(), key=lambda x: x[1]["latency"])

    print(
        f"✓ Best throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.1f} graphs/s)"
    )
    print(f"✓ Best latency: {best_latency[0]} ({best_latency[1]['latency']:.3f} ms/graph)")

    print("\n✅ Demo 5 complete!")
    return results


# ============================================================================
# Main execution
# ============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("SESSION 22 DEMO: PINN Interpretability + GNN Optimization")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  1. PINN Interpretability - Sensitivity analysis")
    print("  2. PINN Interpretability - Physics residual analysis")
    print("  3. PINN Interpretability - Layer activation analysis")
    print("  4. GNN Optimization - Graph Convolutional Network")
    print("  5. GNN Optimization - Architecture comparison")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demos
    results = {}

    try:
        results["demo1"] = demo1_pinn_sensitivity()
        results["demo2"] = demo2_pinn_residual()
        results["demo3"] = demo3_pinn_activations()
        results["demo4"] = demo4_gnn_gcn()
        results["demo5"] = demo5_gnn_comparison()

        # Summary
        print_header("SUMMARY")

        print("\n✅ PINN Interpretability:")
        print(f"  → Sensitivity methods: {results['demo1']['sensitivity_methods']}")
        print(
            f"  → Feature importance: x={results['demo1']['feature_importance']['x']:.3f}, t={results['demo1']['feature_importance']['t']:.3f}"
        )
        print(f"  → Physics residual analysis: {results['demo2']['domain_points']} points")
        print(f"  → Mean residual: {results['demo2']['mean_residual']:.6f}")
        print(f"  → Layer analysis: {results['demo3']['num_layers']} layers")

        print("\n✅ GNN Optimization:")
        print(f"  → Karate Club GCN: {results['demo4']['throughput']:.1f} graphs/s")
        print(f"  → Latency: {results['demo4']['latency']:.3f} ms/graph")
        print(f"  → Architectures tested: {len(results['demo5'])}")

        print("\n" + "=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
