#!/usr/bin/env python3
"""
Driver Health Check & Recommendations
======================================

Comprehensive driver verification for AMD Radeon RX 580 (Polaris).
This script checks kernel drivers, Mesa stack, OpenCL, Vulkan, and ROCm
status, providing recommendations for optimal configuration.

Features:
- Kernel driver (AMDGPU) verification
- Mesa version detection and compatibility check
- OpenCL platform detection (Clover/RustiCL)
- Vulkan driver detection (RADV)
- ROCm availability check (optional)
- Actionable recommendations for issues

Usage:
    python scripts/verify_drivers.py
    python scripts/verify_drivers.py --verbose
    python scripts/verify_drivers.py --json

Version: 1.0.0
Author: Legacy GPU AI Platform Team
License: MIT
"""

import subprocess
import re
import sys
import os
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class DriverReport:
    """Container for comprehensive driver status report"""
    kernel_driver: Optional[Dict[str, str]] = None
    mesa_version: Optional[str] = None
    opencl: Optional[Dict[str, str]] = None
    vulkan: Optional[Dict[str, str]] = None
    rocm: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    overall_status: str = "unknown"
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


def run_command(cmd: List[str], timeout: int = 5) -> Tuple[bool, str, str]:
    """
    Run shell command and capture output.
    
    Args:
        cmd: Command and arguments as list
        timeout: Maximum execution time in seconds
    
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", "Command timed out")
    except FileNotFoundError:
        return (False, "", f"Command not found: {cmd[0]}")
    except Exception as e:
        return (False, "", str(e))


def check_amdgpu_kernel() -> Optional[Dict[str, str]]:
    """
    Check if AMDGPU kernel driver is loaded.
    
    Returns:
        Dictionary with driver info or None if not loaded
    """
    success, stdout, _ = run_command(["lsmod"])
    
    if success and "amdgpu" in stdout:
        # Try to get driver version from modinfo
        success_mod, stdout_mod, _ = run_command(["modinfo", "amdgpu"])
        
        version = "unknown"
        if success_mod:
            version_match = re.search(r'version:\s+([^\n]+)', stdout_mod)
            if version_match:
                version = version_match.group(1).strip()
        
        return {
            "loaded": True,
            "version": version,
            "name": "amdgpu"
        }
    
    return None


def check_mesa_version() -> Optional[str]:
    """
    Detect Mesa version.
    
    Returns:
        Mesa version string or None if not found
    """
    # Try glxinfo first (most reliable)
    success, stdout, _ = run_command(["glxinfo", "-B"])
    
    if success:
        version_match = re.search(r'OpenGL version string:\s+([0-9.]+)', stdout)
        if version_match:
            return version_match.group(1)
    
    # Fallback: check package version
    success, stdout, _ = run_command(["dpkg", "-l", "mesa-common-dev"])
    
    if success:
        version_match = re.search(r'mesa-common-dev\s+([0-9.]+)', stdout)
        if version_match:
            return version_match.group(1)
    
    return None


def check_opencl_status() -> Optional[Dict[str, str]]:
    """
    Check OpenCL availability and platform info.
    
    Returns:
        Dictionary with OpenCL info or None if not available
    """
    success, stdout, _ = run_command(["clinfo", "--list"])
    
    if not success or "Platform" not in stdout:
        return None
    
    # Parse OpenCL platforms
    platforms = []
    current_platform = None
    
    for line in stdout.split('\n'):
        if "Platform #" in line:
            if current_platform:
                platforms.append(current_platform)
            current_platform = {"devices": []}
        elif current_platform is not None:
            if "Platform Name" in line:
                current_platform["name"] = line.split("Platform Name")[-1].strip()
            elif "Platform Version" in line:
                version_str = line.split("Platform Version")[-1].strip()
                version_match = re.search(r'OpenCL\s+([0-9.]+)', version_str)
                if version_match:
                    current_platform["version"] = version_match.group(1)
            elif "`--" in line and "AMD" in line or "Radeon" in line:
                current_platform["devices"].append(line.strip())
    
    if current_platform:
        platforms.append(current_platform)
    
    # Find AMD/Mesa platform
    for platform in platforms:
        name = platform.get("name", "").lower()
        if "clover" in name or "rusticl" in name or "mesa" in name:
            return {
                "available": True,
                "platform": platform.get("name", "Unknown"),
                "version": platform.get("version", "Unknown"),
                "implementation": "Mesa (Clover/RustiCL)"
            }
        elif "amd" in name:
            return {
                "available": True,
                "platform": platform.get("name", "Unknown"),
                "version": platform.get("version", "Unknown"),
                "implementation": "AMD Proprietary"
            }
    
    # OpenCL detected but no AMD platform
    if platforms:
        return {
            "available": False,
            "platform": "None (AMD platform not found)",
            "version": "N/A",
            "implementation": "N/A"
        }
    
    return None


def check_vulkan_status() -> Optional[Dict[str, str]]:
    """
    Check Vulkan availability and driver info.
    
    Returns:
        Dictionary with Vulkan info or None if not available
    """
    success, stdout, _ = run_command(["vulkaninfo", "--summary"])
    
    if not success:
        return None
    
    # Parse Vulkan info
    info = {}
    
    # Check for AMD device
    if "AMD" not in stdout and "Radeon" not in stdout:
        return None
    
    # Extract version
    version_match = re.search(r'Vulkan Instance Version:\s+([0-9.]+)', stdout)
    if version_match:
        info["version"] = version_match.group(1)
    else:
        info["version"] = "Unknown"
    
    # Extract driver name
    driver_match = re.search(r'deviceName\s+=\s+([^\n]+)', stdout)
    if driver_match:
        info["driver"] = driver_match.group(1).strip()
    else:
        info["driver"] = "AMD RADV"
    
    # Check if RADV (Mesa) or AMDVLK
    if "RADV" in stdout or "radv" in stdout.lower():
        info["implementation"] = "Mesa RADV"
    elif "AMDVLK" in stdout:
        info["implementation"] = "AMD AMDVLK"
    else:
        info["implementation"] = "Unknown"
    
    info["available"] = True
    
    return info


def check_rocm_status() -> Optional[Dict[str, Any]]:
    """
    Check ROCm availability (optional for Polaris).
    
    Returns:
        Dictionary with ROCm info or None if not available
    """
    success, stdout, _ = run_command(["rocm-smi", "--showproductname"])
    
    if not success:
        return None
    
    info = {"available": True}
    
    # Try to get version
    success_ver, stdout_ver, _ = run_command(["rocm-smi", "--version"])
    if success_ver:
        version_match = re.search(r'([0-9]+\.[0-9]+\.[0-9]+)', stdout_ver)
        if version_match:
            info["version"] = version_match.group(1)
            
            # Check Polaris support
            major_version = int(info["version"].split('.')[0])
            if major_version >= 6:
                info["polaris_support"] = False
                info["note"] = "ROCm 6.x has limited/no Polaris support"
            elif major_version == 5:
                info["polaris_support"] = True
                info["note"] = "ROCm 5.x supports Polaris (deprecated)"
            else:
                info["polaris_support"] = False
                info["note"] = "Unknown Polaris support"
        else:
            info["version"] = "Unknown"
            info["polaris_support"] = False
    else:
        info["version"] = "Unknown"
        info["polaris_support"] = False
    
    return info


def analyze_and_recommend(report: DriverReport) -> None:
    """
    Analyze driver report and generate recommendations.
    
    Args:
        report: DriverReport object to analyze and update
    """
    recommendations = []
    
    # Check kernel driver
    if not report.kernel_driver:
        recommendations.append(
            "❌ CRITICAL: AMDGPU kernel driver not loaded\n"
            "   Fix: sudo modprobe amdgpu\n"
            "   Make permanent: echo 'amdgpu' | sudo tee -a /etc/modules"
        )
        report.overall_status = "critical"
    
    # Check Mesa version
    if report.mesa_version:
        try:
            mesa_major = int(report.mesa_version.split('.')[0])
            if mesa_major < 22:
                recommendations.append(
                    "⚠️  Mesa version < 22.0 detected (older drivers)\n"
                    "   Recommendation: Update Mesa for better Polaris support\n"
                    "   Fix: sudo apt update && sudo apt upgrade mesa-*"
                )
        except (ValueError, IndexError):
            pass
    else:
        recommendations.append(
            "⚠️  Mesa not detected\n"
            "   Fix: sudo apt install mesa-common-dev libgl1-mesa-dri"
        )
    
    # Check OpenCL
    if not report.opencl or not report.opencl.get("available"):
        recommendations.append(
            "❌ OpenCL not available (REQUIRED for GPU inference)\n"
            "   Fix: sudo apt install mesa-opencl-icd clinfo ocl-icd-opencl-dev opencl-headers\n"
            "   Verify: clinfo --list"
        )
        if report.overall_status != "critical":
            report.overall_status = "error"
    elif report.opencl.get("implementation") == "N/A":
        recommendations.append(
            "⚠️  OpenCL found but AMD platform missing\n"
            "   Fix: sudo apt install mesa-opencl-icd\n"
            "   Verify: clinfo --list"
        )
    
    # Check Vulkan
    if not report.vulkan or not report.vulkan.get("available"):
        recommendations.append(
            "⚠️  Vulkan not detected (RECOMMENDED for best performance)\n"
            "   Fix: sudo apt install mesa-vulkan-drivers vulkan-tools\n"
            "   Verify: vulkaninfo --summary"
        )
    
    # Check ROCm
    if report.rocm and report.rocm.get("available"):
        if not report.rocm.get("polaris_support", False):
            recommendations.append(
                f"⚠️  ROCm {report.rocm.get('version', 'Unknown')} detected with limited Polaris support\n"
                f"   Note: {report.rocm.get('note', 'Consider using OpenCL/Vulkan instead')}\n"
                "   Recommendation: Use OpenCL or Vulkan for better compatibility"
            )
    
    # Set overall status if not already critical/error
    if report.overall_status == "unknown":
        if report.kernel_driver and report.opencl and report.opencl.get("available"):
            report.overall_status = "good"
        elif report.kernel_driver:
            report.overall_status = "warning"
        else:
            report.overall_status = "error"
    
    report.recommendations = recommendations


def print_driver_report(report: DriverReport, verbose: bool = False) -> None:
    """
    Pretty print driver status report.
    
    Args:
        report: DriverReport object to display
        verbose: Show detailed information
    """
    print("=" * 70)
    print("🔧 DRIVER HEALTH CHECK - AMD RX 580 (Polaris)")
    print("=" * 70)
    
    # Kernel driver
    print(f"\n{'[1/5] Kernel Driver (AMDGPU)'}")
    if report.kernel_driver:
        print(f"   ✅ Loaded: Yes")
        print(f"   📦 Version: {report.kernel_driver.get('version', 'unknown')}")
        if verbose:
            print(f"   🔧 Module: {report.kernel_driver.get('name', 'amdgpu')}")
    else:
        print("   ❌ Status: Not loaded - GPU not accessible!")
        print("   💡 Fix: Run 'sudo modprobe amdgpu'")
    
    # Mesa
    print(f"\n{'[2/5] Mesa Graphics Stack'}")
    if report.mesa_version:
        try:
            mesa_major = int(report.mesa_version.split('.')[0])
            print(f"   ✅ Version: {report.mesa_version}")
            if mesa_major >= 22:
                print("   ✅ Status: Good for Polaris (modern version)")
            else:
                print("   ⚠️  Status: Old version - consider updating")
        except (ValueError, IndexError):
            print(f"   ✅ Version: {report.mesa_version}")
    else:
        print("   ❌ Mesa not detected")
    
    # OpenCL
    print(f"\n{'[3/5] OpenCL Compute'}")
    if report.opencl and report.opencl.get("available"):
        print(f"   ✅ Status: Available")
        print(f"   📦 Platform: {report.opencl.get('platform', 'Unknown')}")
        print(f"   🔢 Version: OpenCL {report.opencl.get('version', 'Unknown')}")
        print(f"   🏭 Implementation: {report.opencl.get('implementation', 'Unknown')}")
    else:
        print("   ❌ Status: Not available")
        print("   💡 Required for GPU-accelerated inference")
    
    # Vulkan
    print(f"\n{'[4/5] Vulkan Compute'}")
    if report.vulkan and report.vulkan.get("available"):
        print(f"   ✅ Status: Available")
        print(f"   📦 Driver: {report.vulkan.get('driver', 'Unknown')}")
        print(f"   🔢 Version: Vulkan {report.vulkan.get('version', 'Unknown')}")
        print(f"   🏭 Implementation: {report.vulkan.get('implementation', 'Unknown')}")
        print("   💡 Recommended for best performance")
    else:
        print("   ⚠️  Status: Not available")
        print("   💡 Recommended but not required")
    
    # ROCm
    print(f"\n{'[5/5] ROCm Platform (Optional)'}")
    if report.rocm and report.rocm.get("available"):
        print(f"   ℹ️  Status: Installed")
        print(f"   📦 Version: {report.rocm.get('version', 'Unknown')}")
        
        if report.rocm.get("polaris_support"):
            print("   ✅ Polaris Support: Yes")
        else:
            print("   ⚠️  Polaris Support: Limited/None")
        
        if report.rocm.get("note"):
            print(f"   💡 Note: {report.rocm['note']}")
    else:
        print("   ℹ️  Status: Not installed")
        print("   💡 Optional for RX 580 (OpenCL recommended)")
    
    # Recommendations
    if report.recommendations:
        print(f"\n{'=' * 70}")
        print("📋 RECOMMENDATIONS")
        print("=" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"\n{i}. {rec}")
    
    # Overall status
    print(f"\n{'=' * 70}")
    
    status_emoji = {
        "good": "✅",
        "warning": "⚠️ ",
        "error": "❌",
        "critical": "🚨"
    }
    
    status_message = {
        "good": "DRIVERS ARE OPTIMAL FOR INFERENCE",
        "warning": "DRIVERS FUNCTIONAL BUT IMPROVEMENTS RECOMMENDED",
        "error": "DRIVER ISSUES DETECTED - ACTION REQUIRED",
        "critical": "CRITICAL DRIVER ISSUES - GPU NOT ACCESSIBLE"
    }
    
    emoji = status_emoji.get(report.overall_status, "❓")
    message = status_message.get(report.overall_status, "UNKNOWN STATUS")
    
    print(f"{emoji} {message}")
    
    if report.overall_status == "good":
        print("   Your RX 580 is ready for AI workloads!")
    elif report.overall_status in ["error", "critical"]:
        print("   Follow recommendations above to fix issues")
    
    print("=" * 70)


def check_driver_health() -> DriverReport:
    """
    Perform comprehensive driver health check.
    
    Returns:
        DriverReport object with all checks performed
    """
    report = DriverReport()
    
    # Perform all checks
    print("Running driver diagnostics...\n")
    
    report.kernel_driver = check_amdgpu_kernel()
    report.mesa_version = check_mesa_version()
    report.opencl = check_opencl_status()
    report.vulkan = check_vulkan_status()
    report.rocm = check_rocm_status()
    
    # Analyze and generate recommendations
    analyze_and_recommend(report)
    
    return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AMD RX 580 Driver Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Standard check with recommendations
  %(prog)s --verbose        # Detailed output
  %(prog)s --json           # JSON output for automation
  %(prog)s --json --verbose # JSON with all details

For setup instructions, see: docs/guides/DRIVER_SETUP_RX580.md
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Run checks
    try:
        report = check_driver_health()
        
        if args.json:
            # Output as JSON
            output = asdict(report)
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            print_driver_report(report, verbose=args.verbose)
        
        # Exit code based on status
        exit_codes = {
            "good": 0,
            "warning": 0,
            "error": 1,
            "critical": 2
        }
        
        return exit_codes.get(report.overall_status, 1)
        
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error during check: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
