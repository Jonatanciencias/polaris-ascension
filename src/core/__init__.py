"""
Core compatibility layer.

Provides the historical `core.*` API expected by legacy demos and scripts.
"""

from .gpu import GPUInfo, GPUManager
from .memory import MemoryManager, MemoryStats, MemoryStrategy
from .profiler import Profiler

__all__ = [
    "GPUInfo",
    "GPUManager",
    "MemoryManager",
    "MemoryStats",
    "MemoryStrategy",
    "Profiler",
]
