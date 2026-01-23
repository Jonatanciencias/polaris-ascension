#!/usr/bin/env python3
"""
Power Monitoring Diagnostics
==============================

Diagnoses power monitoring capabilities and provides recommendations.
"""

import os
import glob
import subprocess
import sys


def check_hwmon():
    """Check for hwmon devices."""
    print("\n" + "="*70)
    print("1. Checking Hardware Monitoring Sensors (/sys/class/hwmon/)")
    print("="*70)
    
    hwmon_paths = sorted(glob.glob('/sys/class/hwmon/hwmon*'))
    
    if not hwmon_paths:
        print("❌ No hwmon devices found")
        return False
    
    print(f"✅ Found {len(hwmon_paths)} hwmon device(s):")
    
    amd_found = False
    
    for hwmon in hwmon_paths:
        name_file = os.path.join(hwmon, 'name')
        try:
            with open(name_file) as f:
                name = f.read().strip()
                print(f"\n  📁 {os.path.basename(hwmon)}: {name}")
                
                # Check for power sensor
                power_file = os.path.join(hwmon, 'power1_average')
                if os.path.exists(power_file):
                    try:
                        with open(power_file) as pf:
                            microwatts = int(pf.read().strip())
                            watts = microwatts / 1_000_000
                            print(f"     ⚡ Power: {watts:.2f} W")
                            
                            if 'amdgpu' in name.lower() or 'radeon' in name.lower():
                                print(f"     ✅ AMD GPU DETECTED")
                                amd_found = True
                    except (IOError, ValueError, PermissionError) as e:
                        print(f"     ⚠️  Cannot read power: {e}")
                else:
                    print(f"     ℹ️  No power sensor")
                
                # Check temperature
                temp_file = os.path.join(hwmon, 'temp1_input')
                if os.path.exists(temp_file):
                    try:
                        with open(temp_file) as tf:
                            millicelsius = int(tf.read().strip())
                            celsius = millicelsius / 1000
                            print(f"     🌡️  Temperature: {celsius:.1f} °C")
                    except:
                        pass
                
        except (IOError, PermissionError):
            print(f"  ⚠️  Cannot read {name_file}")
    
    return amd_found


def check_rocm_smi():
    """Check for rocm-smi."""
    print("\n" + "="*70)
    print("2. Checking ROCm SMI")
    print("="*70)
    
    try:
        result = subprocess.run(
            ['rocm-smi', '--showpower'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ rocm-smi is available")
            print("\nOutput:")
            print(result.stdout[:500])
            return True
        else:
            print(f"⚠️  rocm-smi returned error (code {result.returncode})")
            return False
            
    except FileNotFoundError:
        print("❌ rocm-smi not found")
        print("   Install ROCm: https://rocm.docs.amd.com/")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  rocm-smi timed out")
        return False
    except Exception as e:
        print(f"❌ Error running rocm-smi: {e}")
        return False


def check_gpu():
    """Check for GPU."""
    print("\n" + "="*70)
    print("3. Checking GPU Detection")
    print("="*70)
    
    # Check lspci
    try:
        result = subprocess.run(
            ['lspci', '-nn'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split('\n')
        gpu_lines = [l for l in lines if 'VGA' in l or 'Display' in l or '3D' in l]
        
        if gpu_lines:
            print("✅ GPU(s) detected by lspci:")
            for line in gpu_lines:
                print(f"   {line}")
        else:
            print("⚠️  No GPU detected by lspci")
            
    except FileNotFoundError:
        print("⚠️  lspci not found")
    except Exception as e:
        print(f"⚠️  Error running lspci: {e}")
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"\n✅ PyTorch installed: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n⚠️  PyTorch not installed")


def check_permissions():
    """Check file permissions."""
    print("\n" + "="*70)
    print("4. Checking Permissions")
    print("="*70)
    
    test_files = [
        '/sys/class/hwmon/hwmon0/name',
        '/sys/class/hwmon/hwmon1/name'
    ]
    
    readable = 0
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                with open(test_file) as f:
                    f.read()
                print(f"✅ Can read: {test_file}")
                readable += 1
            except PermissionError:
                print(f"❌ Permission denied: {test_file}")
        else:
            print(f"ℹ️  Not found: {test_file}")
    
    if readable > 0:
        print("\n✅ Sensors are readable (no root required)")
    else:
        print("\n⚠️  Cannot read any sensor files")


def provide_recommendations(has_hwmon_amd, has_rocm_smi):
    """Provide recommendations."""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if has_hwmon_amd:
        print("\n✅ EXCELLENT: AMD GPU detected with power sensors")
        print("   Power monitoring should work perfectly!")
        print("\n   Test with:")
        print("   python3 scripts/power_monitor.py --duration 10")
        
    elif has_rocm_smi:
        print("\n✅ GOOD: rocm-smi available (fallback method)")
        print("   Power monitoring will use rocm-smi")
        print("\n   Test with:")
        print("   python3 scripts/power_monitor.py --duration 10")
        
    else:
        print("\n⚠️  NO POWER MONITORING AVAILABLE")
        print("\n   Options:")
        print("\n   1. Install AMD GPU drivers")
        print("      - Ubuntu: sudo apt install amdgpu-dkms")
        print("      - Fedora: sudo dnf install amdgpu-pro-dkms")
        print("\n   2. Install ROCm")
        print("      - https://rocm.docs.amd.com/")
        print("\n   3. Use SIMULATION MODE (for development)")
        print("      - Set environment variable: export POWER_SIMULATE=1")
        print("\n   Note: For academic papers, you NEED real measurements!")


def main():
    """Run diagnostics."""
    print("\n" + "="*70)
    print("   POWER MONITORING DIAGNOSTICS")
    print("   AMD Radeon RX 580 / Polaris GPU")
    print("="*70)
    
    has_hwmon_amd = check_hwmon()
    has_rocm_smi = check_rocm_smi()
    check_gpu()
    check_permissions()
    
    provide_recommendations(has_hwmon_amd, has_rocm_smi)
    
    print("\n" + "="*70)
    print("Diagnostics complete!")
    print("="*70 + "\n")
    
    # Return success if any method available
    return 0 if (has_hwmon_amd or has_rocm_smi) else 1


if __name__ == '__main__':
    sys.exit(main())
