"""
Memory management compatibility layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import psutil  # type: ignore[import-untyped]

from .gpu import GPUManager


class MemoryStrategy(Enum):
    """Memory strategy profile for different VRAM capacities."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class MemoryStats:
    """Runtime memory snapshot."""

    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    ram_percent: float
    gpu_vram_gb: float
    gpu_vram_mb: float


@dataclass
class _Allocation:
    name: str
    size_mb: float
    is_gpu: bool
    persistent: bool
    priority: int


class MemoryManager:
    """
    Legacy-compatible memory manager.

    Tracks simple allocation metadata and provides recommendation helpers.
    """

    def __init__(
        self,
        gpu_vram_mb: Optional[float] = None,
        gpu_vram_gb: Optional[float] = None,
        strategy: Optional[MemoryStrategy] = None,
    ) -> None:
        if gpu_vram_gb is not None:
            self._gpu_vram_mb = float(gpu_vram_gb) * 1024.0
        elif gpu_vram_mb is not None:
            self._gpu_vram_mb = float(gpu_vram_mb)
        else:
            detected = GPUManager().get_info()
            self._gpu_vram_mb = float(detected.vram_mb) if detected else 0.0

        if strategy is None:
            self.strategy = self._select_strategy(self._gpu_vram_mb / 1024.0)
        else:
            self.strategy = strategy

        if self.strategy == MemoryStrategy.CONSERVATIVE:
            self._headroom_mb = 1536
            self._max_single_allocation_percent = 0.25
        elif self.strategy == MemoryStrategy.BALANCED:
            self._headroom_mb = 1024
            self._max_single_allocation_percent = 0.35
        else:
            self._headroom_mb = 768
            self._max_single_allocation_percent = 0.50

        self._allocations: Dict[str, _Allocation] = {}

    @staticmethod
    def _select_strategy(vram_gb: float) -> MemoryStrategy:
        if vram_gb <= 4.0:
            return MemoryStrategy.CONSERVATIVE
        if vram_gb <= 8.0:
            return MemoryStrategy.BALANCED
        return MemoryStrategy.AGGRESSIVE

    def get_stats(self) -> MemoryStats:
        vm = psutil.virtual_memory()
        return MemoryStats(
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
            used_ram_gb=(vm.total - vm.available) / (1024**3),
            ram_percent=float(vm.percent),
            gpu_vram_gb=self._gpu_vram_mb / 1024.0,
            gpu_vram_mb=self._gpu_vram_mb,
        )

    def print_stats(self) -> None:
        stats = self.get_stats()
        print("Memory Statistics")
        print(f"  RAM total: {stats.total_ram_gb:.1f} GB")
        print(f"  RAM available: {stats.available_ram_gb:.1f} GB")
        print(f"  RAM used: {stats.used_ram_gb:.1f} GB ({stats.ram_percent:.1f}%)")
        print(f"  GPU VRAM: {stats.gpu_vram_gb:.1f} GB")
        print(f"  Strategy: {self.strategy.value}")

    def _allocated_gpu_mb(self) -> float:
        return sum(a.size_mb for a in self._allocations.values() if a.is_gpu)

    def detect_memory_pressure(self) -> Tuple[str, float]:
        stats = self.get_stats()
        percent = stats.ram_percent
        if percent < 60:
            return "low", percent
        if percent < 75:
            return "moderate", percent
        if percent < 90:
            return "high", percent
        return "critical", percent

    def can_allocate(self, size_mb: float, use_gpu: bool = True) -> Tuple[bool, str]:
        size_mb = float(size_mb)
        if size_mb <= 0:
            return False, "allocation size must be > 0 MB"

        if not use_gpu or self._gpu_vram_mb <= 0:
            stats = self.get_stats()
            if size_mb < (stats.available_ram_gb * 1024.0 * 0.8):
                return True, "fits in system RAM"
            return False, "insufficient available system RAM"

        allocated = self._allocated_gpu_mb()
        available_gpu = max(self._gpu_vram_mb - self._headroom_mb - allocated, 0.0)
        max_single = self._gpu_vram_mb * self._max_single_allocation_percent

        if size_mb > max_single:
            return False, (
                f"allocation {size_mb:.1f}MB exceeds single-allocation limit "
                f"{max_single:.1f}MB for strategy {self.strategy.value}"
            )
        if size_mb > available_gpu:
            return False, f"insufficient free GPU VRAM (available {available_gpu:.1f}MB)"
        return True, "fits in GPU VRAM"

    def get_recommendations(self, model_size_mb: float) -> Dict[str, object]:
        can_fit_gpu, reason_gpu = self.can_allocate(model_size_mb, use_gpu=True)
        notes: List[str] = []
        if not can_fit_gpu:
            notes.append(reason_gpu)
            notes.append("consider fp16/int8 or batch-size reduction")

        use_quant = model_size_mb > (self._gpu_vram_mb * 0.4) and self._gpu_vram_mb > 0
        use_offload = not can_fit_gpu and self.get_stats().available_ram_gb > 8

        return {
            "strategy": self.strategy.value,
            "use_gpu": can_fit_gpu,
            "use_quantization": use_quant,
            "use_cpu_offload": use_offload,
            "notes": notes,
        }

    def register_allocation(
        self,
        name: str,
        size_mb: float,
        is_gpu: bool = True,
        persistent: bool = False,
        priority: int = 5,
    ) -> bool:
        can_fit, _ = self.can_allocate(size_mb, use_gpu=is_gpu)
        if not can_fit:
            return False
        self._allocations[name] = _Allocation(
            name=name,
            size_mb=float(size_mb),
            is_gpu=is_gpu,
            persistent=persistent,
            priority=int(priority),
        )
        return True

    def free_allocation(self, name: str) -> bool:
        return self._allocations.pop(name, None) is not None
