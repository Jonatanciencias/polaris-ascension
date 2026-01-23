#!/usr/bin/env python3
"""
Post-Reboot Power Sensor Verification
======================================

Verifies that power sensor setup was successful after reboot.
"""

import sys
import os
import glob
import subprocess


def print_header(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def check_kernel_params():
    """Check kernel command line parameters."""
    print_header("1. Kernel Parameters")
    
    try:
        with open('/proc/cmdline') as f:
            cmdline = f.read().strip()
            
        print("📋 Kernel command line:")
        print(f"   {cmdline}\n")
        
        if 'amdgpu.ppfeaturemask' in cmdline:
            print("✅ amdgpu.ppfeaturemask is SET")
            
            # Extract the value
            for param in cmdline.split():
                if 'amdgpu.ppfeaturemask' in param:
                    print(f"   Value: {param}")
            return True
        else:
            print("❌ amdgpu.ppfeaturemask is NOT SET")
            print("   The kernel parameter was not applied.")
            return False
            
    except Exception as e:
        print(f"❌ Error reading kernel parameters: {e}")
        return False


def check_module_loaded():
    """Check if amdgpu module is loaded with correct parameters."""
    print_header("2. AMD GPU Module")
    
    try:
        result = subprocess.run(
            ['lsmod'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'amdgpu' in result.stdout:
            print("✅ amdgpu module is LOADED")
            
            # Try to get module parameters
            param_dir = '/sys/module/amdgpu/parameters'
            if os.path.exists(param_dir):
                print(f"\n📋 Module parameters:")
                
                # Check ppfeaturemask
                ppfm_file = os.path.join(param_dir, 'ppfeaturemask')
                if os.path.exists(ppfm_file):
                    with open(ppfm_file) as f:
                        value = f.read().strip()
                        print(f"   ppfeaturemask: {value}")
                        
                        if value == '-1' or value == '4294967295' or value == '0xffffffff':
                            print("   ✅ Set to full feature mask")
                        else:
                            print(f"   ⚠️  Set to {value} (may not enable all features)")
                
                # Check dpm
                dpm_file = os.path.join(param_dir, 'dpm')
                if os.path.exists(dpm_file):
                    with open(dpm_file) as f:
                        value = f.read().strip()
                        print(f"   dpm: {value} {'✅' if value == '1' else '⚠️'}")
            
            return True
        else:
            print("❌ amdgpu module is NOT LOADED")
            return False
            
    except Exception as e:
        print(f"❌ Error checking module: {e}")
        return False


def check_power_sensors():
    """Check for power sensors."""
    print_header("3. Power Sensors")
    
    sensors_found = False
    amdgpu_sensor_found = False
    
    hwmon_paths = sorted(glob.glob('/sys/class/hwmon/hwmon*'))
    
    if not hwmon_paths:
        print("❌ No hwmon devices found at all")
        return False
    
    print(f"Found {len(hwmon_paths)} hwmon device(s):\n")
    
    for hwmon in hwmon_paths:
        name_file = os.path.join(hwmon, 'name')
        try:
            with open(name_file) as f:
                name = f.read().strip()
            
            is_amdgpu = 'amdgpu' in name.lower() or 'radeon' in name.lower()
            
            print(f"📁 {os.path.basename(hwmon)}: {name}")
            
            # Check for power sensor
            power_file = os.path.join(hwmon, 'power1_average')
            if os.path.exists(power_file):
                try:
                    with open(power_file) as f:
                        microwatts = int(f.read().strip())
                        watts = microwatts / 1_000_000
                    
                    print(f"   ⚡ Power: {watts:.2f} W")
                    sensors_found = True
                    
                    if is_amdgpu:
                        print(f"   ✅✅✅ AMD GPU POWER SENSOR FOUND!")
                        amdgpu_sensor_found = True
                        
                        # Check other power-related files
                        power_cap = os.path.join(hwmon, 'power1_cap')
                        power_cap_max = os.path.join(hwmon, 'power1_cap_max')
                        
                        if os.path.exists(power_cap):
                            with open(power_cap) as f:
                                cap = int(f.read().strip()) / 1_000_000
                            print(f"   📊 Power Cap: {cap:.2f} W")
                        
                        if os.path.exists(power_cap_max):
                            with open(power_cap_max) as f:
                                cap_max = int(f.read().strip()) / 1_000_000
                            print(f"   📊 Max Power Cap: {cap_max:.2f} W")
                        
                except Exception as e:
                    print(f"   ⚠️  Cannot read power: {e}")
            else:
                print(f"   ℹ️  No power sensor")
            
            # Check temperature
            temp_file = os.path.join(hwmon, 'temp1_input')
            if os.path.exists(temp_file):
                try:
                    with open(temp_file) as f:
                        millicelsius = int(f.read().strip())
                        celsius = millicelsius / 1000
                    print(f"   🌡️  Temperature: {celsius:.1f} °C")
                except:
                    pass
            
            print()
            
        except Exception as e:
            print(f"   ⚠️  Error reading device: {e}\n")
    
    return amdgpu_sensor_found


def check_gpu_info():
    """Check GPU information."""
    print_header("4. GPU Information")
    
    try:
        result = subprocess.run(
            ['lspci', '-nn'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = [l for l in result.stdout.split('\n') if 'VGA' in l or 'Display' in l or '3D' in l]
        
        if lines:
            print("🎮 Detected GPUs:")
            for line in lines:
                print(f"   {line}")
        else:
            print("⚠️  No GPU detected by lspci")
            
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")


def run_power_monitor_test():
    """Run a quick power monitor test."""
    print_header("5. Power Monitor Test")
    
    print("🧪 Running quick power monitor test (3 seconds)...\n")
    
    try:
        result = subprocess.run(
            ['python3', 'scripts/power_monitor.py', '--duration', '3', '--verbose'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            if 'kernel_sensors' in result.stdout:
                print("✅✅✅ SUCCESS! Direct power sensor is working!")
                return True
            elif 'estimated' in result.stdout:
                print("⚠️  Still using temperature estimation")
                return False
            else:
                print("⚠️  Unknown method")
                return False
        else:
            print("❌ Power monitor test failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def provide_recommendations(success):
    """Provide recommendations based on results."""
    print_header("Recommendations")
    
    if success:
        print("🎉 CONGRATULATIONS! Direct power sensor is now available!")
        print("")
        print("Next steps:")
        print("  1. Run full benchmark with direct power measurement:")
        print("     python3 scripts/benchmark_all_models_power.py --duration 60")
        print("")
        print("  2. Compare with previous temperature-based results")
        print("")
        print("  3. Note the improved precision (±1W vs ±10-15W)")
        print("")
    else:
        print("❌ Direct power sensor is still not available.")
        print("")
        print("Possible reasons:")
        print("")
        print("1. Driver/Firmware limitation:")
        print("   Some AMD GPUs don't expose power sensors even with correct settings")
        print("   This is a known limitation with certain driver versions")
        print("")
        print("2. Kernel too old:")
        print("   Power sensor support was added in Linux 4.17+")
        print("   Your kernel: $(uname -r)")
        print("   Consider updating to latest kernel")
        print("")
        print("3. Alternative solutions:")
        print("   a) Install ROCm for rocm-smi power monitoring")
        print("      https://rocm.docs.amd.com/")
        print("")
        print("   b) Use external power meter (Kill-A-Watt, etc.)")
        print("      Measures entire system power")
        print("")
        print("   c) Temperature estimation is still valid for academic paper")
        print("      Current method (90-95% correlation) is acceptable")
        print("")


def main():
    """Main verification routine."""
    print("\n" + "="*70)
    print("  AMD GPU Power Sensor - Post-Reboot Verification")
    print("="*70)
    
    results = {
        'kernel_params': False,
        'module_loaded': False,
        'power_sensor': False,
        'monitor_test': False
    }
    
    # Run checks
    results['kernel_params'] = check_kernel_params()
    results['module_loaded'] = check_module_loaded()
    results['power_sensor'] = check_power_sensors()
    check_gpu_info()
    results['monitor_test'] = run_power_monitor_test()
    
    # Final verdict
    print_header("Final Verdict")
    
    success = results['power_sensor'] and results['monitor_test']
    
    print("Checklist:")
    print(f"  {'✅' if results['kernel_params'] else '❌'} Kernel parameters configured")
    print(f"  {'✅' if results['module_loaded'] else '❌'} AMD GPU module loaded")
    print(f"  {'✅' if results['power_sensor'] else '❌'} Power sensor available")
    print(f"  {'✅' if results['monitor_test'] else '❌'} Power monitor working")
    print()
    
    if success:
        print("🎉 STATUS: SUCCESS - Direct power measurement available!")
    elif results['power_sensor']:
        print("⚠️  STATUS: PARTIAL - Sensor found but monitor test failed")
    else:
        print("❌ STATUS: FAILED - Power sensor not available")
    
    print()
    
    provide_recommendations(success)
    
    print("\n" + "="*70)
    print("Verification complete!")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
