"""
Radeon RX 580 AI Framework

A framework for running AI workloads on AMD Radeon RX 580 GPUs.
"""

__version__ = "0.6.0-dev"
__author__ = "RX580 AI Framework Contributors"

from src.core.gpu import GPUManager
from src.core.memory import MemoryManager
from src.utils.config import Config

__all__ = ["GPUManager", "MemoryManager", "Config"]
