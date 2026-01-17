"""
Memory Management Module - Core Layer
=====================================

Professional memory allocation, tracking, and optimization strategies
specifically designed for AMD Polaris GPUs with limited VRAM.

Key Features:
- Intelligent VRAM management for 4GB/8GB cards
- Polaris-specific memory strategies
- Allocation tracking and recommendations
- Automatic memory pressure detection

Version: 0.5.0-dev
License: MIT
"""

import psutil
import time
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies for different VRAM sizes."""
    CONSERVATIVE = "conservative"  # 8GB+ VRAM
    MODERATE = "moderate"          # 6-8GB VRAM
    AGGRESSIVE = "aggressive"      # 4GB VRAM
    MINIMAL = "minimal"            # <4GB VRAM


@dataclass
class MemoryStats:
    """Comprehensive memory statistics container."""
    # System RAM
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float = 0.0
    ram_percent_used: float = 0.0
    
    # GPU VRAM
    total_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    used_vram_gb: float = 0.0
    vram_percent_used: float = 0.0
    
    # Peak usage tracking
    peak_ram_gb: float = 0.0
    peak_vram_gb: float = 0.0
    
    # Allocation count
    num_allocations: int = 0
    
    # Strategy
    strategy: MemoryStrategy = MemoryStrategy.CONSERVATIVE


@dataclass
class AllocationInfo:
    """Information about a memory allocation."""
    name: str
    size_mb: float
    is_gpu: bool
    timestamp: float = 0.0
    persistent: bool = True  # Persistent vs temporary allocation
    priority: int = 5  # 1-10, higher = more important


class MemoryManager:
    """
    Professional memory manager for Polaris GPUs.
    
    This class provides:
    - Smart VRAM allocation with headroom management
    - Memory pressure detection
    - Automatic strategy selection based on hardware
    - Allocation tracking and recommendations
    
    Example:
        manager = MemoryManager(gpu_vram_gb=8.0, strategy=MemoryStrategy.CONSERVATIVE)
        
        if manager.can_allocate(500, use_gpu=True):
            manager.register_allocation("model_weights", 500, is_gpu=True)
    """
    
    def __init__(
        self, 
        gpu_vram_gb: Optional[float] = None,
        strategy: Optional[MemoryStrategy] = None
    ):
        """
        Initialize memory manager with Polaris-specific optimizations.
        
        Args:
            gpu_vram_gb: GPU VRAM in GB (default: 8.0 for RX 580)
            strategy: Memory management strategy (auto-detected if None)
        """
        self.gpu_vram_gb = gpu_vram_gb or 8.0
        
        # Auto-select strategy based on VRAM
        if strategy is None:
            strategy = self._auto_select_strategy()
        self.strategy = strategy
        
        # Initialize stats
        vm = psutil.virtual_memory()
        self._stats = MemoryStats(
            total_ram_gb=vm.total / (1024**3),
            available_ram_gb=vm.available / (1024**3),
            used_ram_gb=vm.used / (1024**3),
            ram_percent_used=vm.percent,
            total_vram_gb=self.gpu_vram_gb,
            available_vram_gb=self.gpu_vram_gb,
            strategy=strategy
        )
        
        # Track allocations
        self._allocations: Dict[str, AllocationInfo] = {}
        
        # Strategy-specific settings
        self._headroom_mb = self._get_headroom_mb()
        self._max_single_allocation_percent = self._get_max_allocation_percent()
    
    def _auto_select_strategy(self) -> MemoryStrategy:
        """Auto-select memory strategy based on VRAM size."""
        if self.gpu_vram_gb >= 8.0:
            return MemoryStrategy.CONSERVATIVE
        elif self.gpu_vram_gb >= 6.0:
            return MemoryStrategy.MODERATE
        elif self.gpu_vram_gb >= 4.0:
            return MemoryStrategy.AGGRESSIVE
        else:
            return MemoryStrategy.MINIMAL
    
    def _get_headroom_mb(self) -> int:
        """Get memory headroom based on strategy."""
        headroom_map = {
            MemoryStrategy.CONSERVATIVE: 1024,  # 1GB headroom
            MemoryStrategy.MODERATE: 768,       # 768MB headroom
            MemoryStrategy.AGGRESSIVE: 512,     # 512MB headroom
            MemoryStrategy.MINIMAL: 256,        # 256MB headroom
        }
        return headroom_map.get(self.strategy, 1024)
    
    def _get_max_allocation_percent(self) -> float:
        """Get max single allocation percentage based on strategy."""
        percent_map = {
            MemoryStrategy.CONSERVATIVE: 0.7,   # 70% max
            MemoryStrategy.MODERATE: 0.6,       # 60% max
            MemoryStrategy.AGGRESSIVE: 0.5,     # 50% max
            MemoryStrategy.MINIMAL: 0.4,        # 40% max
        }
        return percent_map.get(self.strategy, 0.7)
    
    def get_stats(self) -> MemoryStats:
        """
        Get current memory statistics with real-time updates.
        
        Returns:
            MemoryStats object with current memory info
        """
        # Update system RAM stats
        vm = psutil.virtual_memory()
        self._stats.available_ram_gb = vm.available / (1024**3)
        self._stats.used_ram_gb = vm.used / (1024**3)
        self._stats.ram_percent_used = vm.percent
        
        # Calculate used VRAM from allocations
        used_vram_mb = sum(
            alloc.size_mb for alloc in self._allocations.values()
            if alloc.is_gpu
        )
        self._stats.used_vram_gb = used_vram_mb / 1024.0
        self._stats.available_vram_gb = self.gpu_vram_gb - self._stats.used_vram_gb
        self._stats.vram_percent_used = (self._stats.used_vram_gb / self.gpu_vram_gb) * 100
        
        # Update peaks
        self._stats.peak_ram_gb = max(self._stats.peak_ram_gb, self._stats.used_ram_gb)
        self._stats.peak_vram_gb = max(self._stats.peak_vram_gb, self._stats.used_vram_gb)
        
        # Update allocation count
        self._stats.num_allocations = len(self._allocations)
        
        return self._stats
    
    def can_allocate(
        self, 
        size_mb: float, 
        use_gpu: bool = True,
        check_fragmentation: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if memory allocation is possible with detailed reasoning.
        
        Args:
            size_mb: Size to allocate in MB
            use_gpu: Whether to check GPU VRAM (True) or RAM (False)
            check_fragmentation: Check for memory fragmentation issues
            
        Returns:
            Tuple of (can_allocate: bool, reason: str or None)
        """
        if use_gpu:
            # Calculate currently allocated VRAM
            used_vram_mb = sum(
                alloc.size_mb for alloc in self._allocations.values()
                if alloc.is_gpu
            )
            
            available_mb = (self.gpu_vram_gb * 1024) - used_vram_mb - self._headroom_mb
            
            # Check if it fits
            if size_mb > available_mb:
                return False, f"Insufficient VRAM: need {size_mb:.0f}MB, have {available_mb:.0f}MB"
            
            # Check against max single allocation
            max_single_mb = (self.gpu_vram_gb * 1024) * self._max_single_allocation_percent
            if size_mb > max_single_mb:
                return False, f"Allocation too large: {size_mb:.0f}MB exceeds {self.strategy.value} limit of {max_single_mb:.0f}MB"
            
            return True, None
            
        else:
            # Check RAM
            stats = self.get_stats()
            available_gb = stats.available_ram_gb - 2.0  # 2GB system headroom
            size_gb = size_mb / 1024.0
            
            if size_gb > available_gb:
                return False, f"Insufficient RAM: need {size_gb:.1f}GB, have {available_gb:.1f}GB"
            
            return True, None
    
    def register_allocation(
        self, 
        name: str, 
        size_mb: float, 
        is_gpu: bool = True,
        persistent: bool = True,
        priority: int = 5
    ) -> bool:
        """
        Register a memory allocation with metadata.
        
        Args:
            name: Identifier for the allocation
            size_mb: Size in MB
            is_gpu: Whether this is GPU memory
            persistent: Whether allocation persists across operations
            priority: Allocation priority (1=highest, 10=lowest)
            
        Returns:
            True if registration successful
        """
        # Check if allocation is possible
        can_alloc, reason = self.can_allocate(size_mb, use_gpu=is_gpu)
        if not can_alloc:
            logger.warning(f"Cannot register allocation '{name}': {reason}")
            return False
        
        # Create allocation info
        alloc = AllocationInfo(
            name=name,
            size_mb=size_mb,
            is_gpu=is_gpu,
            timestamp=time.time(),
            persistent=persistent,
            priority=priority
        )
        
        key = f"{'gpu' if is_gpu else 'cpu'}_{name}"
        self._allocations[key] = alloc
        
        logger.debug(f"Registered {'GPU' if is_gpu else 'CPU'} allocation: {name} ({size_mb:.0f}MB)")
        return True
    
    def free_allocation(self, name: str, is_gpu: bool = True) -> bool:
        """
        Free a registered memory allocation.
        
        Args:
            name: Identifier for the allocation
            is_gpu: Whether this is GPU memory
            
        Returns:
            True if freed successfully
        """
        key = f"{'gpu' if is_gpu else 'cpu'}_{name}"
        if key in self._allocations:
            alloc = self._allocations[key]
            del self._allocations[key]
            logger.debug(f"Freed {'GPU' if is_gpu else 'CPU'} allocation: {name} ({alloc.size_mb:.0f}MB)")
            return True
        return False
    
    def clear_allocations(self, gpu_only: bool = False, non_persistent_only: bool = True):
        """
        Clear tracked allocations.
        
        Args:
            gpu_only: Only clear GPU allocations
            non_persistent_only: Only clear non-persistent allocations
        """
        to_remove = []
        
        for key, alloc in self._allocations.items():
            # Filter by GPU/CPU
            if gpu_only and not alloc.is_gpu:
                continue
            
            # Filter by persistence
            if non_persistent_only and alloc.persistent:
                continue
            
            to_remove.append(key)
        
        for key in to_remove:
            del self._allocations[key]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} allocations")
    
    def detect_memory_pressure(self) -> Tuple[str, float]:
        """
        Detect current memory pressure level.
        
        Returns:
            Tuple of (level: str, percent_used: float)
            Levels: 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
        """
        stats = self.get_stats()
        vram_percent = stats.vram_percent_used
        
        if vram_percent < 50:
            return 'LOW', vram_percent
        elif vram_percent < 70:
            return 'MODERATE', vram_percent
        elif vram_percent < 85:
            return 'HIGH', vram_percent
        else:
            return 'CRITICAL', vram_percent
    
    def get_recommendations(self, model_size_mb: float) -> Dict[str, any]:
        """
        Get Polaris-specific memory optimization recommendations.
        
        Args:
            model_size_mb: Model size in MB
            
        Returns:
            Dictionary with strategy-aware recommendations
        """
        stats = self.get_stats()
        pressure_level, pressure_percent = self.detect_memory_pressure()
        
        recommendations = {
            'use_gpu': False,
            'use_quantization': False,
            'use_cpu_offload': False,
            'max_batch_size': 1,
            'strategy': self.strategy.value,
            'current_pressure': pressure_level,
            'notes': []
        }
        
        # Check if model fits in VRAM with current strategy
        can_fit, reason = self.can_allocate(model_size_mb, use_gpu=True)
        
        if can_fit:
            recommendations['use_gpu'] = True
            
            # Calculate safe batch size based on strategy
            available_mb = (self.gpu_vram_gb * 1024) - model_size_mb - self._headroom_mb
            
            # Batch overhead estimation (20-30% per item)
            batch_overhead = model_size_mb * 0.25
            if batch_overhead > 0:
                recommendations['max_batch_size'] = max(1, int(available_mb / batch_overhead))
            
            recommendations['notes'].append(
                f"✓ Model fits in VRAM ({self.strategy.value} strategy)"
            )
            recommendations['notes'].append(
                f"✓ Safe batch size: {recommendations['max_batch_size']}"
            )
        else:
            # Model doesn't fit - suggest strategy-specific optimizations
            recommendations['notes'].append(
                f"✗ Model too large: {reason}"
            )
            
            # Try INT8 quantization (50% size)
            quantized_size = model_size_mb * 0.5
            can_fit_q8, _ = self.can_allocate(quantized_size, use_gpu=True)
            
            if can_fit_q8:
                recommendations['use_quantization'] = True
                recommendations['use_gpu'] = True
                recommendations['notes'].append(
                    "→ INT8 quantization should enable GPU inference"
                )
            else:
                # Try INT4 (25% size) for aggressive strategies
                quantized_size_int4 = model_size_mb * 0.25
                can_fit_q4, _ = self.can_allocate(quantized_size_int4, use_gpu=True)
                
                if can_fit_q4 and self.strategy in [MemoryStrategy.AGGRESSIVE, MemoryStrategy.MINIMAL]:
                    recommendations['use_quantization'] = True
                    recommendations['use_gpu'] = True
                    recommendations['notes'].append(
                        "→ INT4 quantization required (quality may degrade)"
                    )
                else:
                    # Need CPU offloading
                    recommendations['use_cpu_offload'] = True
                    recommendations['notes'].append(
                        "→ CPU offloading required for model layers"
                    )
                    recommendations['notes'].append(
                        "→ Consider smaller model variant"
                    )
        
        # Add strategy-specific advice
        if self.strategy == MemoryStrategy.MINIMAL:
            recommendations['notes'].append(
                "⚠ MINIMAL strategy active - consider upgrading GPU for better performance"
            )
        elif pressure_level == 'HIGH' or pressure_level == 'CRITICAL':
            recommendations['notes'].append(
                f"⚠ Memory pressure {pressure_level} ({pressure_percent:.1f}% used)"
            )
        
        return recommendations
    
    def print_stats(self):
        """Print comprehensive memory statistics with strategy info"""
        stats = self.get_stats()
        pressure_level, pressure_percent = self.detect_memory_pressure()
        
        print("\n╔════════════════════════════════════════╗")
        print("║       MEMORY STATISTICS                ║")
        print("╠════════════════════════════════════════╣")
        print(f"║ Strategy: {self.strategy.value:<27} ║")
        print(f"║ Pressure: {pressure_level:<13} ({pressure_percent:5.1f}%)  ║")
        print("╠════════════════════════════════════════╣")
        print(f"║ GPU VRAM (Polaris):                    ║")
        print(f"║   Total:     {self.gpu_vram_gb:6.2f} GB                 ║")
        print(f"║   Used:      {stats.used_vram_gb:6.2f} GB ({stats.vram_percent_used:5.1f}%)        ║")
        print(f"║   Available: {stats.available_vram_gb:6.2f} GB                 ║")
        print(f"║   Peak:      {stats.peak_vram_gb:6.2f} GB                 ║")
        print(f"║   Headroom:  {self._headroom_mb:6.0f} MB                 ║")
        print("╠════════════════════════════════════════╣")
        print(f"║ System RAM:                            ║")
        print(f"║   Total:     {stats.total_ram_gb:6.2f} GB                 ║")
        print(f"║   Used:      {stats.used_ram_gb:6.2f} GB ({stats.ram_percent_used:5.1f}%)        ║")
        print(f"║   Available: {stats.available_ram_gb:6.2f} GB                 ║")
        print("╠════════════════════════════════════════╣")
        print(f"║ Allocations: {stats.num_allocations:<26} ║")
        print("╚════════════════════════════════════════╝\n")
