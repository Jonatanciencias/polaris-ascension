"""
Sparse Networks Demo - Magnitude & Structured Pruning
=====================================================

Session 10: Comprehensive demonstration of sparse operations:
1. Magnitude pruning (unstructured)
2. Structured channel pruning
3. Gradual pruning with polynomial schedule
4. Sparse vs dense matmul benchmark
5. Memory reduction measurements

Usage:
    python examples/demo_sparse.py
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compute.sparse import (
    GradualPruner,
    MagnitudePruner,
    SparseOperations,
    StructuredPruner,
    create_sparse_layer,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_magnitude_pruning():
    """Demo 1: Magnitude-based pruning on CNN weights."""
    print_section("DEMO 1: Magnitude Pruning (Unstructured)")

    # Simulate Conv2D weights: (64 out_channels, 32 in_channels, 3x3 kernel)
    np.random.seed(42)
    weights = np.random.randn(64, 32, 3, 3) * 0.1

    print(f"\nOriginal weights shape: {weights.shape}")
    print(f"Total parameters: {weights.size:,}")

    # Test different sparsity levels
    sparsity_levels = [0.5, 0.7, 0.9, 0.95]

    print("\n" + "-" * 70)
    print("Sparsity | Params Kept | Compression | Threshold")
    print("-" * 70)

    for sparsity in sparsity_levels:
        pruner = MagnitudePruner(sparsity=sparsity)
        pruned_weights, mask = pruner.prune_layer(weights)

        actual_sparsity = pruner.measure_sparsity(pruned_weights)
        params_kept = np.count_nonzero(mask)
        compression = weights.size / params_kept if params_kept > 0 else 0
        threshold = pruner.pruning_history[-1]["threshold"]

        print(f"{sparsity:6.1%}   | {params_kept:11,} | {compression:10.2f}x | {threshold:9.6f}")

    # Detailed analysis at 90% sparsity
    print("\n" + "-" * 70)
    print("Detailed Analysis at 90% Sparsity:")
    print("-" * 70)

    pruner = MagnitudePruner(sparsity=0.9)
    pruned_weights, mask = pruner.prune_layer(weights)

    print(f"  Original size:     {weights.nbytes:,} bytes")
    print(f"  Sparse size (CSR): ~{int(weights.nbytes * 0.1):,} bytes (estimated)")
    print(f"  Memory reduction:  {(1 - 0.1) * 100:.1f}%")
    print(f"  Remaining params:  {np.count_nonzero(mask):,}")
    print(f"  Weight range:      [{pruned_weights.min():.6f}, {pruned_weights.max():.6f}]")

    # Visualize sparsity pattern (first channel)
    print("\n  Sparsity pattern (first 8x8 of channel 0):")
    channel_0 = mask[0, 0, :, :]  # First output channel, first input channel
    # Repeat to make it 8x8
    pattern = np.kron(channel_0, np.ones((3, 3)))[:8, :8]
    for row in pattern:
        print("  " + "".join("█" if x else "░" for x in row))


def demo_structured_pruning():
    """Demo 2: Structured channel pruning comparison."""
    print_section("DEMO 2: Structured Channel Pruning")

    # Simulate larger Conv2D layer
    np.random.seed(42)
    weights = np.random.randn(128, 64, 3, 3) * 0.1

    print(f"\nOriginal weights shape: {weights.shape}")
    print(f"  Output channels: {weights.shape[0]}")
    print(f"  Input channels:  {weights.shape[1]}")
    print(f"  Total params:    {weights.size:,}")

    # Compare L1 vs L2 importance metrics
    print("\n" + "-" * 70)
    print("Importance Metric | Channels Kept | Params Kept | Reduction")
    print("-" * 70)

    for metric in ["l1", "l2"]:
        pruner = StructuredPruner(sparsity=0.5, importance_metric=metric, granularity="channel")
        pruned_weights, kept_indices = pruner.prune_channels(weights)

        reduction = 1 - (pruned_weights.size / weights.size)
        print(
            f"{metric.upper():17s} | {len(kept_indices):13} | {pruned_weights.size:11,} | {reduction:8.1%}"
        )

    # Detailed L1 channel pruning
    print("\n" + "-" * 70)
    print("Channel Importance Scores (L1 norm) - Top 10:")
    print("-" * 70)

    pruner = StructuredPruner(sparsity=0.5, importance_metric="l1")

    # Compute scores manually for visualization
    scores = np.sum(np.abs(weights), axis=(1, 2, 3))
    top_10_idx = np.argsort(scores)[-10:][::-1]

    for rank, idx in enumerate(top_10_idx, 1):
        print(f"  #{rank:2}: Channel {idx:3} - Score: {scores[idx]:.4f}")

    # Prune and compare shapes
    pruned_weights, kept_indices = pruner.prune_channels(weights)

    print(f"\n  Original shape: {weights.shape}")
    print(f"  Pruned shape:   {pruned_weights.shape}")
    print(f"  Speedup:        ~{weights.shape[0] / pruned_weights.shape[0]:.2f}x (channels)")
    print(
        f"  Memory:         {pruned_weights.nbytes:,} bytes ({pruned_weights.nbytes / weights.nbytes:.1%})"
    )


def demo_gradual_pruning():
    """Demo 3: Gradual pruning schedule visualization."""
    print_section("DEMO 3: Gradual Pruning with Polynomial Schedule")

    # Create gradual pruner
    pruner = GradualPruner(
        initial_sparsity=0.0,
        final_sparsity=0.9,
        begin_step=1000,
        end_step=10000,
        frequency=100,
        pruning_method="magnitude",
    )

    print("\nGradual Pruning Configuration:")
    print(f"  Initial sparsity: {pruner.initial_sparsity:.0%}")
    print(f"  Final sparsity:   {pruner.final_sparsity:.0%}")
    print(f"  Begin step:       {pruner.begin_step:,}")
    print(f"  End step:         {pruner.end_step:,}")
    print(f"  Frequency:        Every {pruner.frequency} steps")
    print(f"  Schedule:         Polynomial decay (cubic)")

    # Generate and display schedule
    schedule = pruner.get_schedule(num_steps=12000)

    print("\n" + "-" * 70)
    print("Step     | Target Sparsity | Prune?")
    print("-" * 70)

    key_steps = [0, 1000, 2000, 3000, 5000, 7000, 9000, 10000, 11000]
    for step in key_steps:
        sparsity = pruner.compute_sparsity(step)
        should_prune = "YES" if pruner.should_prune(step) else "NO"
        print(f"{step:7,} | {sparsity:14.1%} | {should_prune}")

    # ASCII visualization of schedule
    print("\n" + "-" * 70)
    print("Schedule Visualization (0% → 90% sparsity):")
    print("-" * 70)
    print("\n  Progress: [" + "." * 50 + "]")
    print("            0%                   50%                  100%\n")

    for step in range(0, 11000, 500):
        sparsity = pruner.compute_sparsity(step)
        progress = (step - pruner.begin_step) / (pruner.end_step - pruner.begin_step)
        progress = max(0, min(1, progress))

        bar_length = int(progress * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)

        print(f"  Step {step:5,}: [{bar}] {sparsity:5.1%}")

    # Simulate pruning steps
    print("\n" + "-" * 70)
    print("Simulated Pruning (sample weights):")
    print("-" * 70)

    np.random.seed(42)
    weights = np.random.randn(100, 100)

    for step in [1000, 3000, 5000, 7000, 9000, 10000]:
        pruned_weights, mask = pruner.prune_step(weights.copy(), step)
        actual_sparsity = pruner.base_pruner.measure_sparsity(pruned_weights)
        print(
            f"  Step {step:5,}: {actual_sparsity:5.1%} sparse ({np.count_nonzero(mask):,} params kept)"
        )


def demo_sparse_matmul_benchmark():
    """Demo 4: Sparse vs dense matmul benchmark."""
    print_section("DEMO 4: Sparse vs Dense Matrix Multiplication Benchmark")

    ops = SparseOperations(gpu_family="polaris")

    # Test different matrix sizes and sparsity levels
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    sparsities = [0.5, 0.7, 0.9]

    print("\nBenchmarking matrix multiplication...\n")
    print("-" * 70)
    print("Size          | Sparsity | Dense Time | Sparse Time | Speedup")
    print("-" * 70)

    for size in sizes:
        for sparsity in sparsities:
            # Create sparse matrix
            np.random.seed(42)
            A = np.random.randn(*size)
            mask = np.random.rand(*size) > sparsity
            A_sparse = A * mask

            B = np.random.randn(size[1], 256)

            # Benchmark dense
            start = time.perf_counter()
            for _ in range(10):
                _ = np.matmul(A_sparse, B)
            dense_time = (time.perf_counter() - start) / 10

            # Current implementation is placeholder (dense fallback)
            # In real GPU implementation, sparse would be faster
            sparse_time = dense_time  # Placeholder
            theoretical_speedup = 1.0 / (1.0 - sparsity) if sparsity > 0.5 else 1.0

            print(
                f"{size[0]}x{size[1]:4} | {sparsity:7.0%} | {dense_time*1000:9.3f}ms | {sparse_time*1000:10.3f}ms | {theoretical_speedup:6.2f}x*"
            )

    print("-" * 70)
    print("* Theoretical speedup (GPU implementation planned for v0.6.0)")
    print("  Current version uses dense fallback")

    # CSR format analysis
    print("\n" + "-" * 70)
    print("CSR Format Analysis:")
    print("-" * 70)

    np.random.seed(42)
    matrix = np.random.randn(1000, 1000)
    matrix[matrix < 0.5] = 0  # Make 69% sparse

    result = ops.analyze_sparsity(matrix)
    print(f"\n  Matrix size:         {matrix.shape}")
    print(f"  Sparsity:            {result['sparsity']:.1%}")
    print(f"  Nonzero elements:    {result['nonzero_elements']:,}")
    print(f"  Potential speedup:   {result['potential_speedup']:.1f}x")
    print(f"  Recommendation:      {result['recommendation']}")
    print(f"  Wavefront aligned:   {result['wavefront_aligned']}")

    # Convert to CSR
    csr = ops.to_csr(matrix[:100, :100])  # Smaller for display
    print(f"\n  CSR format (100x100 subset):")
    print(f"    Values array:      {len(csr['values']):,} elements")
    print(f"    Column indices:    {len(csr['col_indices']):,} elements")
    print(f"    Row pointers:      {len(csr['row_pointers']):,} elements")

    dense_bytes = 100 * 100 * 8  # float64
    sparse_bytes = (
        len(csr["values"]) * 8  # values
        + len(csr["col_indices"]) * 4  # int32 indices
        + len(csr["row_pointers"]) * 4
    )  # int32 pointers

    print(f"    Dense size:        {dense_bytes:,} bytes")
    print(f"    CSR size:          {sparse_bytes:,} bytes")
    print(f"    Compression:       {dense_bytes / sparse_bytes:.2f}x")


def demo_memory_reduction():
    """Demo 5: Memory reduction measurement."""
    print_section("DEMO 5: Memory Reduction with Pruning")

    # Simulate a small neural network
    print("\nSimulated Neural Network:")
    print("  3-layer MLP: 1024 → 512 → 256 → 10")

    layers = {
        "layer1": np.random.randn(512, 1024),
        "layer2": np.random.randn(256, 512),
        "layer3": np.random.randn(10, 256),
    }

    total_params = sum(w.size for w in layers.values())
    total_memory = sum(w.nbytes for w in layers.values())

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Total memory:     {total_memory:,} bytes ({total_memory / 1024:.1f} KB)")

    # Prune at different sparsity levels
    print("\n" + "-" * 70)
    print("Sparsity | Params Kept | Memory Kept | Compression")
    print("-" * 70)

    for sparsity in [0.5, 0.7, 0.9, 0.95]:
        pruner = MagnitudePruner(sparsity=sparsity, scope="global")
        pruned_weights, masks = pruner.prune_model(layers)

        stats = pruner.get_compression_stats(masks)

        # Estimate sparse memory (CSR format)
        sparse_memory = 0
        for name in layers:
            nonzero = np.count_nonzero(masks[name])
            sparse_memory += (
                nonzero * 8  # values (float64)
                + nonzero * 4  # col_indices (int32)
                + layers[name].shape[0] * 4
            )  # row_pointers (int32)

        compression = total_memory / sparse_memory if sparse_memory > 0 else 0

        print(
            f"{sparsity:7.0%} | {stats['remaining_parameters']:11,} | {sparse_memory:11,} | {compression:10.2f}x"
        )

    # Layer-wise analysis
    print("\n" + "-" * 70)
    print("Layer-wise Pruning (90% sparsity):")
    print("-" * 70)

    pruner = MagnitudePruner(sparsity=0.9, scope="local")
    pruned_weights, masks = pruner.prune_model(layers)

    print(f"\n{'Layer':<10} | {'Original':<12} | {'Remaining':<12} | {'Sparsity':<10}")
    print("-" * 70)

    for name in layers:
        original = layers[name].size
        remaining = np.count_nonzero(masks[name])
        sparsity = pruner.measure_sparsity(pruned_weights[name])
        print(f"{name:<10} | {original:12,} | {remaining:12,} | {sparsity:9.1%}")

    total_remaining = sum(np.count_nonzero(masks[name]) for name in layers)
    total_sparsity = 1 - (total_remaining / total_params)

    print("-" * 70)
    print(f"{'Total':<10} | {total_params:12,} | {total_remaining:12,} | {total_sparsity:9.1%}")


def main():
    """Run all demos."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SPARSE NETWORKS DEMONSTRATION" + " " * 24 + "║")
    print("║" + " " * 68 + "║")
    print("║" + " " * 10 + "Session 10: Magnitude & Structured Pruning" + " " * 15 + "║")
    print("║" + " " * 20 + "AMD Polaris RX 580" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        demo_magnitude_pruning()
        demo_structured_pruning()
        demo_gradual_pruning()
        demo_sparse_matmul_benchmark()
        demo_memory_reduction()

        print("\n" + "=" * 70)
        print("  All demos completed successfully!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • Magnitude pruning: Simple, effective, 90% sparsity achievable")
        print("  • Structured pruning: GPU-friendly, no sparse kernels needed")
        print("  • Gradual pruning: Better accuracy preservation than one-shot")
        print("  • Memory reduction: 5-10x compression at high sparsity")
        print("  • CSR format: Efficient sparse matrix storage")
        print("\nNext Steps:")
        print("  • Implement GPU-accelerated sparse kernels (v0.6.0)")
        print("  • Add fine-tuning after pruning")
        print("  • Benchmark on real models (ResNet, BERT)")
        print("  • Integrate with quantization for hybrid compression")
        print()

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
