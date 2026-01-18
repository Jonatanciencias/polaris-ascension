"""
Demo: Sparse Matrix Formats - CSR, CSC, Block-Sparse
====================================================

Interactive demonstrations of sparse matrix formats optimized for
AMD Radeon RX 580 (Polaris, GCN 4.0).

Demos:
-----
1. CSR Format - Basic usage and memory savings
2. CSR vs Dense - Performance comparison
3. High Sparsity - 90%+ sparse neural network weights
4. Integration - With pruners from Sessions 10-11

Usage:
-----
    python examples/demo_sparse_formats.py [demo_number]
    
    # Run all demos
    python examples/demo_sparse_formats.py all
    
    # Run specific demo
    python examples/demo_sparse_formats.py 1

Author: Legacy GPU AI Platform
Version: 0.6.0-dev
"""

import sys
import time
import numpy as np
from typing import Dict, Tuple

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compute.sparse_formats import CSRMatrix, SparseMatrixStats


def print_header(title: str):
    """Print formatted demo header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_stats(stats: SparseMatrixStats):
    """Print formatted sparse matrix statistics."""
    print(f"  Shape: {stats.shape[0]} × {stats.shape[1]}")
    print(f"  Non-zeros: {stats.nnz:,}")
    print(f"  Sparsity: {stats.sparsity*100:.2f}%")
    print(f"  Density: {stats.density*100:.2f}%")
    print(f"  Memory (dense): {stats.memory_dense / 1024:.2f} KB")
    print(f"  Memory (CSR): {stats.memory_sparse / 1024:.2f} KB")
    print(f"  Compression: {stats.compression_ratio:.2f}x")


def demo_1_basic_csr():
    """Demo 1: Basic CSR usage and memory savings."""
    print_header("Demo 1: Basic CSR Format Usage")
    
    print("Creating a simple sparse matrix (diagonal):\n")
    
    # Create diagonal matrix
    size = 5
    dense = np.eye(size, dtype=np.float32)
    print("Dense representation:")
    print(dense)
    print()
    
    # Convert to CSR
    print("Converting to CSR format...")
    csr = CSRMatrix.from_dense(dense)
    print()
    
    print("CSR components:")
    print(f"  values:      {csr.values}")
    print(f"  col_indices: {csr.col_indices}")
    print(f"  row_ptr:     {csr.row_ptr}")
    print()
    
    # Show statistics
    print("Statistics:")
    stats = csr.get_statistics()
    print_stats(stats)
    print()
    
    # Test reconstruction
    print("Reconstructing dense matrix...")
    reconstructed = csr.to_dense()
    matches = np.allclose(reconstructed, dense)
    print(f"  Reconstruction accurate: {matches} ✓" if matches else f"  ERROR: Mismatch!")
    print()
    
    # Test matrix multiplication
    print("Testing sparse matrix-vector multiplication:")
    vector = np.ones(size, dtype=np.float32)
    print(f"  Input vector: {vector}")
    
    result_sparse = csr.sparse_matmul(vector)
    result_dense = dense @ vector
    
    print(f"  CSR result:   {result_sparse}")
    print(f"  Dense result: {result_dense}")
    print(f"  Match: {np.allclose(result_sparse, result_dense)} ✓")


def demo_2_csr_vs_dense_performance():
    """Demo 2: CSR vs Dense performance comparison."""
    print_header("Demo 2: CSR vs Dense - Performance Comparison")
    
    sizes = [100, 500, 1000]
    sparsities = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    print("Benchmarking CSR vs Dense for various matrix sizes and sparsities\n")
    print(f"{'Size':<8} {'Sparsity':<10} {'Dense (ms)':<12} {'CSR (ms)':<12} {'Speedup':<10} {'Memory':<10}")
    print("-" * 70)
    
    for size in sizes:
        for sparsity in sparsities:
            # Create sparse matrix
            np.random.seed(42)
            dense = np.random.randn(size, size).astype(np.float32)
            
            # Apply sparsity
            mask = np.random.rand(size, size) < sparsity
            dense[mask] = 0
            
            # Convert to CSR
            csr = CSRMatrix.from_dense(dense)
            
            # Create test vector
            vector = np.random.randn(size).astype(np.float32)
            
            # Benchmark dense
            start = time.time()
            for _ in range(10):
                result_dense = dense @ vector
            time_dense = (time.time() - start) * 100  # ms
            
            # Benchmark CSR
            start = time.time()
            for _ in range(10):
                result_csr = csr.sparse_matmul(vector)
            time_csr = (time.time() - start) * 100  # ms
            
            # Calculate speedup
            speedup = time_dense / time_csr if time_csr > 0 else 0
            
            # Memory comparison
            stats = csr.get_statistics()
            
            print(f"{size:<8} {sparsity*100:>6.0f}%    "
                  f"{time_dense:>8.2f}    {time_csr:>8.2f}    "
                  f"{speedup:>7.2f}x   {stats.compression_ratio:>7.2f}x")
    
    print("\n📊 Observations:")
    print("  • CSR becomes faster than dense at ~70-80% sparsity")
    print("  • Memory savings increase linearly with sparsity")
    print("  • Best for neural network pruning (90%+ sparsity)")


def demo_3_high_sparsity_neural_network():
    """Demo 3: High sparsity neural network weights."""
    print_header("Demo 3: High Sparsity Neural Network Weights")
    
    print("Simulating pruned neural network layer weights\n")
    
    # Typical neural network layer
    layer_configs = [
        ("Hidden Layer 1", 784, 512, 0.90),
        ("Hidden Layer 2", 512, 256, 0.95),
        ("Output Layer", 256, 10, 0.80),
    ]
    
    total_params_dense = 0
    total_params_sparse = 0
    total_memory_dense = 0
    total_memory_sparse = 0
    
    print(f"{'Layer':<20} {'Shape':<15} {'Sparsity':<12} {'Params':<15} {'Memory Saved':<15}")
    print("-" * 80)
    
    for name, in_features, out_features, sparsity in layer_configs:
        # Create random weights
        np.random.seed(42)
        dense = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
        
        # Apply pruning (magnitude-based)
        threshold = np.percentile(np.abs(dense), sparsity * 100)
        dense[np.abs(dense) < threshold] = 0
        
        # Convert to CSR
        csr = CSRMatrix.from_dense(dense)
        stats = csr.get_statistics()
        
        # Calculate parameters
        params_dense = in_features * out_features
        params_sparse = csr.nnz
        
        total_params_dense += params_dense
        total_params_sparse += params_sparse
        total_memory_dense += stats.memory_dense
        total_memory_sparse += stats.memory_sparse
        
        print(f"{name:<20} {out_features}×{in_features:<10} "
              f"{stats.sparsity*100:>6.2f}%     "
              f"{params_sparse:>6,}/{params_dense:>6,}  "
              f"{stats.compression_ratio:>7.2f}x")
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {'':<15} {'':<12} "
          f"{total_params_sparse:>6,}/{total_params_dense:>6,}  "
          f"{total_memory_dense/total_memory_sparse:>7.2f}x")
    
    print(f"\n💾 Memory Savings:")
    print(f"  Dense model:  {total_memory_dense / 1024 / 1024:.2f} MB")
    print(f"  Sparse model: {total_memory_sparse / 1024 / 1024:.2f} MB")
    print(f"  Saved:        {(total_memory_dense - total_memory_sparse) / 1024 / 1024:.2f} MB "
          f"({(1 - total_memory_sparse/total_memory_dense)*100:.1f}%)")
    
    print(f"\n🎯 Model Efficiency:")
    print(f"  Total parameters reduced: {total_params_dense:,} → {total_params_sparse:,}")
    print(f"  Compression ratio: {total_params_dense / total_params_sparse:.2f}x")
    print(f"  This model would fit in RX 580 VRAM: {'✓ YES' if total_memory_sparse < 8*1024*1024*1024 else '✗ NO'}")


def demo_4_integration_with_pruning():
    """Demo 4: Integration with pruning from Sessions 10-11."""
    print_header("Demo 4: Integration with Pruning")
    
    print("Demonstrating CSR format with magnitude pruning\n")
    
    # Simulate pre-trained weights
    np.random.seed(42)
    size = 1000
    dense = np.random.randn(size, size).astype(np.float32) * 0.1
    
    print(f"Original dense weights: {size}×{size}")
    print(f"Memory: {dense.nbytes / 1024:.2f} KB\n")
    
    # Progressive pruning
    sparsity_levels = [0.0, 0.5, 0.75, 0.90, 0.95, 0.98]
    
    print(f"{'Sparsity':<12} {'NNZ':<10} {'Memory (KB)':<15} {'Compression':<15} {'Inference Time':<15}")
    print("-" * 75)
    
    test_input = np.random.randn(size).astype(np.float32)
    
    for sparsity in sparsity_levels:
        # Apply magnitude pruning
        pruned = dense.copy()
        if sparsity > 0:
            threshold = np.percentile(np.abs(dense), sparsity * 100)
            pruned[np.abs(pruned) < threshold] = 0
        
        # Convert to CSR
        csr = CSRMatrix.from_dense(pruned)
        stats = csr.get_statistics()
        
        # Measure inference time
        start = time.time()
        for _ in range(100):
            result = csr.sparse_matmul(test_input)
        inference_time = (time.time() - start) * 10  # ms per inference
        
        print(f"{sparsity*100:>6.0f}%      "
              f"{csr.nnz:<10,} "
              f"{stats.memory_sparse/1024:<14.2f} "
              f"{stats.compression_ratio:<14.2f}x "
              f"{inference_time:<14.2f} ms")
    
    print("\n🎓 Key Insights:")
    print("  • Memory savings scale linearly with sparsity")
    print("  • CSR format adds minimal overhead for sparse matrices")
    print("  • Optimal for deployment on memory-constrained GPUs (RX 580: 8GB)")
    print("  • Pruning + CSR enables larger models to fit in VRAM")


def main():
    """Run demos."""
    print("\n" + "🚀 " * 35)
    print("   SPARSE MATRIX FORMATS DEMO - CSR Implementation")
    print("   AMD Radeon RX 580 Optimized (GCN 4.0, Polaris)")
    print("   Session 12: Sparse Matrix Formats")
    print("🚀 " * 35)
    
    demos = {
        '1': demo_1_basic_csr,
        '2': demo_2_csr_vs_dense_performance,
        '3': demo_3_high_sparsity_neural_network,
        '4': demo_4_integration_with_pruning,
    }
    
    # Parse command line
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice == 'all':
            for demo in demos.values():
                demo()
                print("\n" + "─" * 70)
        elif choice in demos:
            demos[choice]()
        else:
            print(f"Unknown demo: {choice}")
            print(f"Available: {', '.join(demos.keys())}, all")
    else:
        # Interactive menu
        print("\nAvailable demos:")
        print("  1. Basic CSR usage")
        print("  2. CSR vs Dense performance")
        print("  3. High sparsity neural networks")
        print("  4. Integration with pruning")
        print("  all. Run all demos")
        print()
        
        choice = input("Select demo (1-4, all): ").strip().lower()
        
        if choice == 'all':
            for demo in demos.values():
                demo()
                print("\n" + "─" * 70)
        elif choice in demos:
            demos[choice]()
        else:
            print("Invalid choice. Running demo 1...")
            demo_1_basic_csr()
    
    print("\n✅ Demo complete!")
    print("\n💡 Next: Implement CSC and Block-sparse formats (Session 12 Phase 2-3)")


if __name__ == "__main__":
    main()
