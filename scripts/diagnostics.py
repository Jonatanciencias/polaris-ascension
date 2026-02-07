#!/usr/bin/env python3
"""
System Diagnostics Script

Comprehensive system diagnostics for RX 580 AI setup.
"""

import sys
import os
import subprocess
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verify_hardware import detect_amd_gpu, rocm_available


def run_command(cmd, description):
    """Run a shell command and return output"""
    print(f"\n### {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False


def main():
    """Main diagnostics function"""
    print("=" * 70)
    print("Radeon RX 580 AI Framework - System Diagnostics")
    print("=" * 70)
    
    # Section 1: Hardware Detection
    print("\n" + "=" * 70)
    print("1. HARDWARE DETECTION")
    print("=" * 70)
    
    run_command(["lspci", "-v", "-s", "03:00.0"], "GPU PCI Information")
    run_command(["uname", "-a"], "Kernel Information")
    
    # Section 2: Driver Status
    print("\n" + "=" * 70)
    print("2. DRIVER STATUS")
    print("=" * 70)
    
    run_command(["lsmod"], "Loaded Kernel Modules (filtered for amdgpu)")
    os.system("lsmod | grep -i amd")
    
    # Section 3: OpenCL
    print("\n" + "=" * 70)
    print("3. OPENCL STATUS")
    print("=" * 70)
    
    if run_command(["clinfo"], "OpenCL Platform Information"):
        print("✅ OpenCL is working")
    else:
        print("⚠️  OpenCL not available or not working")
    
    # Section 4: ROCm
    print("\n" + "=" * 70)
    print("4. ROCM STATUS")
    print("=" * 70)
    
    if run_command(["rocm-smi"], "ROCm System Management"):
        print("✅ ROCm is available")
    else:
        print("ℹ️  ROCm not installed (optional for RX 580)")
    
    # Section 5: Python Environment
    print("\n" + "=" * 70)
    print("5. PYTHON ENVIRONMENT")
    print("=" * 70)
    
    run_command([sys.executable, "--version"], "Python Version")
    run_command([sys.executable, "-m", "pip", "list"], "Installed Packages")
    
    # Section 6: Framework Detection
    print("\n" + "=" * 70)
    print("6. FRAMEWORK STATUS")
    print("=" * 70)

    gpu = detect_amd_gpu()
    if gpu:
        print("\n### GPU Detection")
        print(f"   Name: {gpu.name}")
        print(f"   Vendor: {gpu.vendor}")
        print(f"   Driver: {gpu.driver}")
        print(f"   OpenCL: {gpu.opencl_version}")
        print(f"   VRAM: {gpu.vram_gb:.1f} GB")
        print("   Backend: OPENCL")
    else:
        print("\n### GPU Detection")
        print("   ⚠️  No AMD GPU detected via OpenCL")
        print("   Backend: CPU fallback")

    print("\n### Runtime Availability")
    print(f"   ROCm available: {'yes' if rocm_available() else 'no'}")

    print("\n### Memory Statistics")
    vm = psutil.virtual_memory()
    print(f"   Total RAM: {vm.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {vm.available / (1024**3):.1f} GB")
    print(f"   RAM utilization: {vm.percent:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nReview the output above to identify any issues.")
    print("For common problems and solutions, see docs/troubleshooting.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
