#!/usr/bin/env python3
"""
Hardware Verification Script

Detects and verifies AMD Radeon RX 580 GPU configuration.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.gpu import GPUManager
from src.core.memory import MemoryManager


def main():
    """Main verification function"""
    print("=" * 60)
    print("Radeon RX 580 Hardware Verification")
    print("=" * 60)
    
    # Initialize GPU manager
    gpu_manager = GPUManager()
    
    # Detect GPU
    print("\n[1/3] Detecting GPU...")
    gpu_info = gpu_manager.detect_gpu()
    
    if not gpu_info:
        print("❌ No AMD Radeon RX 580 detected!")
        print("\nPlease ensure:")
        print("  - Your GPU is properly installed")
        print("  - AMDGPU drivers are loaded")
        print("  - You have necessary permissions")
        return 1
    
    print(f"✅ GPU Detected: {gpu_info.name}")
    print(f"   PCI ID: {gpu_info.pci_id}")
    print(f"   Architecture: {gpu_info.architecture}")
    print(f"   Driver: {gpu_info.driver or 'Unknown'}")
    
    # Check compute capabilities
    print("\n[2/3] Checking compute capabilities...")
    
    backend = gpu_manager.get_compute_backend()
    print(f"   Recommended backend: {backend.upper()}")
    
    if gpu_info.opencl_available:
        print("   ✅ OpenCL: Available")
    else:
        print("   ⚠️  OpenCL: Not available")
        print("      Install: apt install ocl-icd-opencl-dev opencl-headers clinfo")
    
    if gpu_info.rocm_available:
        print("   ✅ ROCm: Available")
    else:
        print("   ⚠️  ROCm: Not available (optional)")
        print("      Note: ROCm may have limited support for Polaris GPUs")
    
    # Check memory
    print("\n[3/3] Checking memory...")
    memory_manager = MemoryManager(gpu_info.vram_mb)
    stats = memory_manager.get_stats()
    
    print(f"   System RAM: {stats.total_ram_gb:.1f} GB")
    print(f"   Available RAM: {stats.available_ram_gb:.1f} GB")
    print(f"   GPU VRAM: {stats.gpu_vram_gb:.1f} GB")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if stats.total_ram_gb < 16:
        print("⚠️  System RAM < 16GB: May limit model offloading capabilities")
    else:
        print("✅ System RAM sufficient for CPU offloading")
    
    if not gpu_info.opencl_available and not gpu_info.rocm_available:
        print("❌ No compute backend available!")
        print("   Next steps:")
        print("   1. Install OpenCL: ./scripts/setup.sh")
        print("   2. Rerun verification: python scripts/verify_hardware.py")
        return 1
    
    print("\n✅ System is ready for AI workloads!")
    print("\nNext steps:")
    print("  1. Run setup: ./scripts/setup.sh")
    print("  2. Run diagnostics: python scripts/diagnostics.py")
    print("  3. Try an example: python examples/simple_inference.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
