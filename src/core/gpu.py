"""
GPU detection and hardware hints for AMD Polaris-class devices.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, cast

try:
    import pyopencl as cl

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    cl = cast(Any, None)


@dataclass
class GPUInfo:
    """Hardware information for one GPU device."""

    name: str
    vendor: str
    platform: str
    driver: str
    opencl_version: str
    vram_bytes: int
    architecture: str = "Polaris"
    pci_id: Optional[str] = None
    opencl_available: bool = True
    rocm_available: bool = False

    @property
    def vram_mb(self) -> int:
        return int(self.vram_bytes / (1024**2))

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024**3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "platform": self.platform,
            "driver": self.driver,
            "opencl_version": self.opencl_version,
            "vram_mb": self.vram_mb,
            "vram_gb": self.vram_gb,
            "architecture": self.architecture,
            "pci_id": self.pci_id,
            "opencl_available": self.opencl_available,
            "rocm_available": self.rocm_available,
        }

    def items(self) -> Iterator[Tuple[str, object]]:
        """Dictionary-like compatibility helper for legacy demos."""
        return iter(self.to_dict().items())


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


def _detect_rocm() -> bool:
    return Path("/opt/rocm").exists() or shutil.which("rocminfo") is not None


class GPUManager:
    """Legacy-compatible GPU manager with OpenCL-based detection."""

    def __init__(self) -> None:
        self._info: Optional[GPUInfo] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize manager and detect AMD GPU if present."""
        self._info = self.detect_gpu()
        self._initialized = True
        return self._info is not None

    def is_initialized(self) -> bool:
        return self._initialized

    def detect_gpu(self) -> Optional[GPUInfo]:
        """Detect AMD GPU from available OpenCL platforms."""
        if not HAS_OPENCL:
            return None

        candidates: list[GPUInfo] = []
        for platform in cl.get_platforms():
            for device in platform.get_devices(device_type=cl.device_type.GPU):
                name = cast(str, device.get_info(cl.device_info.NAME)).strip()
                vendor = cast(str, device.get_info(cl.device_info.VENDOR)).strip()
                if not _is_amd_radeon(name, vendor):
                    continue

                info = GPUInfo(
                    name=name,
                    vendor=vendor,
                    platform=platform.name.strip(),
                    driver=cast(str, device.get_info(cl.device_info.DRIVER_VERSION)).strip(),
                    opencl_version=cast(str, device.get_info(cl.device_info.VERSION)).strip(),
                    vram_bytes=int(cast(Any, device.get_info(cl.device_info.GLOBAL_MEM_SIZE))),
                    architecture="Polaris" if "polaris" in name.lower() else "Unknown",
                    opencl_available=True,
                    rocm_available=_detect_rocm(),
                )
                candidates.append(info)

        if not candidates:
            return None

        candidates.sort(key=lambda item: _priority(item.name))
        self._info = candidates[0]
        return self._info

    def get_info(self) -> Optional[GPUInfo]:
        """Return detected GPU info, running detection if needed."""
        if self._info is None and not self._initialized:
            self.initialize()
        return self._info

    def get_compute_backend(self) -> str:
        """Return recommended compute backend."""
        info = self.get_info()
        if info and info.opencl_available:
            return "opencl"
        return "cpu"

    def get_optimization_hints(self) -> Dict[str, object]:
        """
        Return conservative hardware-aware optimization hints.
        """
        info = self.get_info()
        if info is None:
            return {
                "backend": "cpu",
                "recommended_precision": "fp32",
                "recommended_batch_size": 1,
                "tile_size": 16,
                "notes": "No AMD OpenCL GPU detected.",
            }

        batch = 4 if info.vram_gb >= 8 else 2
        return {
            "backend": "opencl",
            "recommended_precision": "fp16" if info.vram_gb >= 8 else "fp32",
            "recommended_batch_size": batch,
            "tile_size": 16,
            "workgroup_size": 256,
            "architecture": info.architecture,
            "notes": "Tune with real benchmarks for production workloads.",
        }
