#!/usr/bin/env python3
"""
Demo: Core Layer Professional Implementation
==============================================

Demonstrates the enhanced Core Layer capabilities:
- Multi-method GPU detection (lspci/rocm-smi/opencl)
- GCN-specific optimizations and hints
- Polaris memory strategies
- Memory pressure detection
- Professional hardware reporting

Version: 0.5.0-dev
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.gpu import GPUManager
from core.memory import MemoryManager, MemoryStrategy


def demo_gpu_detection():
    """Demonstrate enhanced GPU detection"""
    print("=" * 60)
    print("DEMO 1: Enhanced GPU Detection")
    print("=" * 60)
    
    gpu = GPUManager()
    
    # Attempt initialization - shows professional output
    gpu.initialize()
    
    # Get comprehensive GPU info
    if gpu.is_initialized():
        print("\n--- Comprehensive GPU Information ---")
        info = gpu.get_info()
        
        for key, value in info.items():
            print(f"  {key:.<35} {value}")
        
        # Get GCN optimization hints
        print("\n--- GCN Optimization Hints ---")
        hints = gpu.get_optimization_hints()
        for key, value in hints.items():
            print(f"  {key:.<35} {value}")
    else:
        print("\n⚠ GPU initialization failed (may be expected in CI)")


def demo_memory_strategies():
    """Demonstrate Polaris memory strategies"""
    print("\n\n" + "=" * 60)
    print("DEMO 2: Polaris Memory Strategies")
    print("=" * 60)
    
    # Test different VRAM configurations
    configs = [
        (4.0, "RX 580 4GB"),
        (8.0, "RX 580 8GB"),
        (6.5, "Hypothetical 6.5GB"),
    ]
    
    for vram_gb, label in configs:
        print(f"\n--- Configuration: {label} ---")
        mem = MemoryManager(gpu_vram_gb=vram_gb)
        
        print(f"Auto-selected strategy: {mem.strategy.value}")
        print(f"Headroom reserved: {mem._headroom_mb}MB")
        print(f"Max single allocation: {mem._max_single_allocation_percent*100:.0f}%")
        
        # Check memory pressure
        pressure, percent = mem.detect_memory_pressure()
        print(f"Current pressure: {pressure} ({percent:.1f}%)")


def demo_memory_recommendations():
    """Demonstrate intelligent memory recommendations"""
    print("\n\n" + "=" * 60)
    print("DEMO 3: Intelligent Memory Recommendations")
    print("=" * 60)
    
    # Assume 8GB Polaris
    mem = MemoryManager(gpu_vram_gb=8.0)
    mem.print_stats()
    
    # Test various model sizes
    test_models = [
        (512, "Tiny model (MobileNet)"),
        (2048, "Medium model (ResNet50)"),
        (4096, "Large model (BERT-base)"),
        (7000, "XL model (GPT2-medium)"),
        (15000, "XXL model (Too large)"),
    ]
    
    print("\n--- Model Fit Analysis ---")
    for size_mb, description in test_models:
        print(f"\n{description} ({size_mb}MB):")
        
        can_fit, reason = mem.can_allocate(size_mb, use_gpu=True)
        if can_fit:
            print(f"  ✓ Fits in VRAM")
        else:
            print(f"  ✗ {reason}")
        
        recs = mem.get_recommendations(size_mb)
        print(f"  Strategy: {recs['strategy']}")
        print(f"  Use GPU: {recs['use_gpu']}")
        print(f"  Quantization: {recs['use_quantization']}")
        print(f"  CPU Offload: {recs['use_cpu_offload']}")
        
        if recs['notes']:
            for note in recs['notes']:
                print(f"    {note}")


def demo_allocation_tracking():
    """Demonstrate allocation tracking with priorities"""
    print("\n\n" + "=" * 60)
    print("DEMO 4: Allocation Tracking & Priorities")
    print("=" * 60)
    
    mem = MemoryManager(gpu_vram_gb=8.0)
    
    # Register various allocations
    allocations = [
        ("resnet50_model", 2048, True, 1),  # High priority, persistent
        ("inference_buffer", 512, False, 5),  # Medium priority, temporary
        ("cache_layer1", 256, True, 7),  # Low priority
    ]
    
    print("\n--- Registering Allocations ---")
    for name, size_mb, persistent, priority in allocations:
        success = mem.register_allocation(
            name, size_mb, is_gpu=True, 
            persistent=persistent, priority=priority
        )
        status = "✓" if success else "✗"
        print(f"{status} {name}: {size_mb}MB (priority={priority}, persistent={persistent})")
    
    # Show stats
    mem.print_stats()
    
    # Check pressure
    pressure, percent = mem.detect_memory_pressure()
    print(f"\nMemory Pressure: {pressure} ({percent:.1f}%)")
    
    # Cleanup
    print("\n--- Freeing Allocations ---")
    mem.free_allocation("cache_layer1")
    mem.print_stats()


def main():
    """Run all Core Layer demos"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  CORE LAYER PROFESSIONAL DEMO".center(58) + "║")
    print("║" + "  Polaris GPU + GCN Optimizations".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        demo_gpu_detection()
        demo_memory_strategies()
        demo_memory_recommendations()
        demo_allocation_tracking()
        
        print("\n\n" + "=" * 60)
        print("DEMO COMPLETE ✓")
        print("=" * 60)
        print("\nCore Layer Features Demonstrated:")
        print("  ✓ Multi-method GPU detection")
        print("  ✓ GCN optimization hints")
        print("  ✓ Polaris memory strategies")
        print("  ✓ Memory pressure detection")
        print("  ✓ Intelligent recommendations")
        print("  ✓ Priority-based allocation tracking")
        
    except Exception as e:
        print(f"\n\n⚠ Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
