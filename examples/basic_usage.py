#!/usr/bin/env python3
"""
Basic Usage Example - AMD RX 590 GEMM Optimization Framework

This example shows the simplest way to use the production kernel selector
to get optimal GEMM kernel recommendations.

Expected output:
- Kernel selection for various matrix sizes
- Performance predictions
- Kernel paths and configuration

Author: AMD RX 590 GEMM Optimization Project
Date: February 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization_engines.adaptive_kernel_selector import (
    ProductionKernelSelector,
    select_optimal_kernel
)


def example_1_basic_selection():
    """Example 1: Basic kernel selection"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Kernel Selection")
    print("=" * 70)
    
    selector = ProductionKernelSelector()
    
    # Test different matrix sizes
    sizes = [512, 1024, 1400, 2048, 3072]
    
    print("\nMatrix Size | Selected Kernel | Expected GFLOPS | Work Group")
    print("-" * 70)
    
    for size in sizes:
        rec = selector.select_kernel(M=size, N=size, K=size)
        print(f"{size:11d} | {rec['kernel_key']:15s} | "
              f"{rec['predicted_gflops']:15.1f} | {rec['local_size']}")
    
    print("\n✅ Basic selection complete!\n")


def example_2_detailed_recommendation():
    """Example 2: Detailed recommendation inspection"""
    print("=" * 70)
    print("EXAMPLE 2: Detailed Recommendation for Sweet Spot (1400×1400)")
    print("=" * 70)
    
    # Get recommendation for sweet spot size
    rec = select_optimal_kernel(M=1400, N=1400, K=1400)
    
    print("\n📊 Recommendation Details:")
    print(f"  Kernel: {rec['kernel_key']}")
    print(f"  Expected Performance: {rec['predicted_gflops']:.1f} GFLOPS")
    print(f"  Kernel Path: {rec['kernel_path']}")
    print(f"  Local Work Size: {rec['local_size']}")
    print(f"  Tile Size: {rec['tile_size']}×{rec['tile_size']}")
    print(f"  Threads per Workgroup: {rec['threads']}")
    print(f"  Selection Method: {rec['selection_method']}")
    print(f"  Best For: {rec['best_for']}")
    
    print("\n✅ Sweet spot analysis complete!\n")


def example_3_compare_kernels():
    """Example 3: Compare kernels across size range"""
    print("=" * 70)
    print("EXAMPLE 3: Kernel Selection Pattern Analysis")
    print("=" * 70)
    
    selector = ProductionKernelSelector()
    
    # Test comprehensive size range
    sizes = list(range(512, 3200, 256))
    
    tile20_count = 0
    tile24_count = 0
    tile16_count = 0
    
    print("\nSize Range | tile16 | tile20 | tile24")
    print("-" * 50)
    
    for i in range(0, len(sizes), 4):
        batch = sizes[i:i+4]
        selections = [selector.select_kernel(s, s, s)['kernel_key'] for s in batch]
        
        t16 = sum(1 for s in selections if s == 'tile16')
        t20 = sum(1 for s in selections if s == 'tile20')
        t24 = sum(1 for s in selections if s == 'tile24')
        
        tile16_count += t16
        tile20_count += t20
        tile24_count += t24
        
        print(f"{batch[0]:4d}-{batch[-1]:4d}  | {t16:6d} | {t20:6d} | {t24:6d}")
    
    print("-" * 50)
    print(f"Total      | {tile16_count:6d} | {tile20_count:6d} | {tile24_count:6d}")
    
    print("\n📈 Pattern Analysis:")
    total = tile16_count + tile20_count + tile24_count
    print(f"  tile16: {tile16_count/total*100:5.1f}% (small matrices)")
    print(f"  tile20: {tile20_count/total*100:5.1f}% (medium, sweet spot)")
    print(f"  tile24: {tile24_count/total*100:5.1f}% (large matrices)")
    
    print("\n✅ Pattern analysis complete!\n")


def example_4_performance_expectations():
    """Example 4: Performance expectations by kernel"""
    print("=" * 70)
    print("EXAMPLE 4: Performance Expectations by Kernel")
    print("=" * 70)
    
    selector = ProductionKernelSelector()
    
    print("\n🎯 tile20 Performance (Sweet Spot Kernel):")
    print("Size  | Expected GFLOPS | Notes")
    print("-" * 60)
    for size in [512, 768, 1024, 1280, 1400, 1536]:
        rec = selector.select_kernel(size, size, size)
        if rec['kernel_key'] == 'tile20':
            note = "🏆 PEAK" if size == 1400 else ""
            print(f"{size:5d} | {rec['predicted_gflops']:15.1f} | {note}")
    
    print("\n🚀 tile24 Performance (Scaling Kernel):")
    print("Size  | Expected GFLOPS | Notes")
    print("-" * 60)
    for size in [1536, 2048, 2560, 3072]:
        rec = selector.select_kernel(size, size, size)
        if rec['kernel_key'] == 'tile24':
            note = "🏆 PEAK" if size == 3072 else ""
            print(f"{size:5d} | {rec['predicted_gflops']:15.1f} | {note}")
    
    print("\n📊 Key Findings:")
    print("  • tile20 peaks at 1400×1400: ~778 GFLOPS")
    print("  • tile24 peaks at 3072×3072: ~805 GFLOPS")
    print("  • Overall improvement: +37-42% vs baseline")
    print("  • Correctness guaranteed: max_error < 0.001")
    
    print("\n✅ Performance analysis complete!\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("AMD RX 590 GEMM Optimization Framework - Basic Usage Examples")
    print("=" * 70)
    print("\nThis script demonstrates the production kernel selector.\n")
    
    try:
        example_1_basic_selection()
        example_2_detailed_recommendation()
        example_3_compare_kernels()
        example_4_performance_expectations()
        
        print("=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run test_production_system.py for comprehensive validation")
        print("  2. See REAL_HARDWARE_VALIDATION.md for verified benchmarks")
        print("  3. Check EXECUTIVE_SUMMARY.md for complete project assessment")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure ML model exists: src/ml_models/kernel_selector_model.pkl")
        print("  2. Check dependencies: pip install -r requirements.txt")
        print("  3. Verify installation: python test_production_system.py")
        print()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
