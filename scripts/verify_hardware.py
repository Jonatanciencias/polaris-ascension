#!/usr/bin/env python3
"""
Hardware verification for AMD Polaris systems (RX 580/RX 590 family).

This script uses OpenCL directly, so it does not depend on legacy modules that
were removed from the current source tree.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import psutil

try:
    import pyopencl as cl

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    cl = None  # type: ignore[assignment]


@dataclass
class OpenCLGPU:
    name: str
    vendor: str
    platform: str
    driver: str
    opencl_version: str
    vram_bytes: int

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024**3)


def _is_amd_radeon(name: str, vendor: str) -> bool:
    text = f"{name} {vendor}".lower()
    return "amd" in text or "radeon" in text or "advanced micro devices" in text


def _priority(name: str) -> int:
    lname = name.lower()
    if "rx 580" in lname or "rx580" in lname:
        return 0
    if "rx 590" in lname or "rx590" in lname:
        return 1
    if "polaris" in lname:
        return 2
    return 3


def detect_amd_gpu() -> Optional[OpenCLGPU]:
    """Detect the best AMD GPU candidate through OpenCL."""
    if not HAS_OPENCL:
        return None

    candidates: List[OpenCLGPU] = []
    for platform in cl.get_platforms():  # type: ignore[union-attr]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for device in devices:
            name = device.get_info(cl.device_info.NAME).strip()
            vendor = device.get_info(cl.device_info.VENDOR).strip()
            if not _is_amd_radeon(name, vendor):
                continue

            driver = device.get_info(cl.device_info.DRIVER_VERSION).strip()
            version = device.get_info(cl.device_info.VERSION).strip()
            vram_bytes = int(device.get_info(cl.device_info.GLOBAL_MEM_SIZE))

            candidates.append(
                OpenCLGPU(
                    name=name,
                    vendor=vendor,
                    platform=platform.name.strip(),
                    driver=driver,
                    opencl_version=version,
                    vram_bytes=vram_bytes,
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda gpu: _priority(gpu.name))
    return candidates[0]


def rocm_available() -> bool:
    """Best-effort ROCm detection."""
    return Path("/opt/rocm").exists() or shutil.which("rocminfo") is not None


def main() -> int:
    print("=" * 60)
    print("Radeon RX 580 Hardware Verification")
    print("=" * 60)

    print("\n[1/3] Detecting GPU...")
    gpu = detect_amd_gpu()

    if gpu is None:
        print("FAIL: No AMD Radeon GPU detected through OpenCL.")
        print("\nPlease ensure:")
        print("  - GPU is properly installed")
        print("  - OpenCL runtime is installed and visible")
        print("  - User has permissions to access /dev/dri")
        print("  - Run: clinfo | rg -n 'Platform|Device'")
        return 1

    print(f"OK GPU Detected: {gpu.name}")
    print(f"   Vendor: {gpu.vendor}")
    print(f"   Platform: {gpu.platform}")
    print(f"   Driver: {gpu.driver}")
    print(f"   OpenCL: {gpu.opencl_version}")

    print("\n[2/3] Checking compute capabilities...")
    print("   Recommended backend: OPENCL")
    print("   OK OpenCL: Available")

    if rocm_available():
        print("   OK ROCm: Available")
    else:
        print("   WARN ROCm: Not available (optional for Polaris)")

    print("\n[3/3] Checking memory...")
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    print(f"   System RAM: {total_ram_gb:.1f} GB")
    print(f"   Available RAM: {available_ram_gb:.1f} GB")
    print(f"   GPU VRAM: {gpu.vram_gb:.1f} GB")

    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)

    if total_ram_gb < 16:
        print("WARN System RAM < 16GB: CPU offloading can be constrained.")
    else:
        print("OK System RAM is sufficient for typical offloading.")

    if "rx 580" not in gpu.name.lower() and "rx 590" not in gpu.name.lower():
        print("WARN Non RX 580/590 detected; performance profiles may vary.")

    print("\nOK System is ready for AI workloads on OpenCL.")
    print("\nNext steps:")
    print("  1. Run diagnostics: ./venv/bin/python scripts/diagnostics.py")
    print("  2. Run validation: ./venv/bin/python test_production_system.py")
    print("  3. Run test suite: ./venv/bin/pytest tests/ -v")

    return 0


if __name__ == "__main__":
    sys.exit(main())
