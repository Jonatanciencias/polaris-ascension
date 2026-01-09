"""
Memory Management Module

Handles memory allocation, tracking, and optimization strategies.
"""

import psutil
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MemoryStats:
    """Memory statistics container"""
    total_ram_gb: float
    available_ram_gb: float
    gpu_vram_gb: Optional[float] = None
    used_ram_gb: float = 0.0
    used_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    peak_vram_gb: float = 0.0


class MemoryManager:
    """Manages memory allocation and optimization"""
    
    def __init__(self, gpu_vram_mb: Optional[int] = None):
        """
        Initialize memory manager.
        
        Args:
            gpu_vram_mb: GPU VRAM in MB (default: 8192 for RX 580)
        """
        self.gpu_vram_mb = gpu_vram_mb or 8192
        self._stats = MemoryStats(
            total_ram_gb=psutil.virtual_memory().total / (1024**3),
            available_ram_gb=psutil.virtual_memory().available / (1024**3),
            gpu_vram_gb=self.gpu_vram_mb / 1024.0
        )
        self._allocations: Dict[str, int] = {}
    
    def get_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            MemoryStats object with current memory info
        """
        vm = psutil.virtual_memory()
        self._stats.available_ram_gb = vm.available / (1024**3)
        self._stats.used_ram_gb = vm.used / (1024**3)
        
        return self._stats
    
    def can_allocate(self, size_mb: int, use_gpu: bool = True) -> bool:
        """
        Check if memory allocation is possible.
        
        Args:
            size_mb: Size to allocate in MB
            use_gpu: Whether to check GPU VRAM (True) or RAM (False)
            
        Returns:
            True if allocation is possible
        """
        if use_gpu:
            available = self.gpu_vram_mb - sum(
                size for name, size in self._allocations.items() 
                if name.startswith('gpu_')
            )
            # Leave 512MB headroom for system
            return size_mb <= (available - 512)
        else:
            stats = self.get_stats()
            # Leave 2GB headroom for system
            return size_mb / 1024.0 <= (stats.available_ram_gb - 2.0)
    
    def register_allocation(self, name: str, size_mb: int, is_gpu: bool = True):
        """
        Register a memory allocation.
        
        Args:
            name: Identifier for the allocation
            size_mb: Size in MB
            is_gpu: Whether this is GPU memory
        """
        key = f"{'gpu' if is_gpu else 'cpu'}_{name}"
        self._allocations[key] = size_mb
        
        if is_gpu:
            self._stats.used_vram_gb = sum(
                size for name, size in self._allocations.items() 
                if name.startswith('gpu_')
            ) / 1024.0
            self._stats.peak_vram_gb = max(
                self._stats.peak_vram_gb,
                self._stats.used_vram_gb
            )
    
    def unregister_allocation(self, name: str, is_gpu: bool = True):
        """
        Unregister a memory allocation.
        
        Args:
            name: Identifier for the allocation
            is_gpu: Whether this is GPU memory
        """
        key = f"{'gpu' if is_gpu else 'cpu'}_{name}"
        if key in self._allocations:
            del self._allocations[key]
            
            if is_gpu:
                self._stats.used_vram_gb = sum(
                    size for name, size in self._allocations.items() 
                    if name.startswith('gpu_')
                ) / 1024.0
    
    def get_recommendations(self, model_size_mb: int) -> Dict[str, any]:
        """
        Get memory optimization recommendations.
        
        Args:
            model_size_mb: Model size in MB
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'use_gpu': False,
            'use_quantization': False,
            'use_cpu_offload': False,
            'max_batch_size': 1,
            'notes': []
        }
        
        # Check if model fits in VRAM
        if self.can_allocate(model_size_mb, use_gpu=True):
            recommendations['use_gpu'] = True
            
            # Calculate safe batch size
            available_vram = self.gpu_vram_mb - model_size_mb - 512
            # Rough estimate: each batch item needs ~20% of model size
            batch_overhead = model_size_mb * 0.2
            if batch_overhead > 0:
                recommendations['max_batch_size'] = max(1, int(available_vram / batch_overhead))
            
            recommendations['notes'].append(
                f"Model fits in VRAM. Safe batch size: {recommendations['max_batch_size']}"
            )
        else:
            # Model doesn't fit, suggest optimizations
            recommendations['notes'].append(
                "Model too large for VRAM. Optimizations recommended:"
            )
            
            # Check if 8-bit quantization would help
            quantized_size = model_size_mb * 0.5
            if self.can_allocate(quantized_size, use_gpu=True):
                recommendations['use_quantization'] = True
                recommendations['use_gpu'] = True
                recommendations['notes'].append(
                    "- 8-bit quantization should allow GPU inference"
                )
            else:
                # Need CPU offloading
                recommendations['use_cpu_offload'] = True
                recommendations['notes'].append(
                    "- Use CPU offloading for parts of the model"
                )
                recommendations['notes'].append(
                    "- Consider 4-bit quantization if available"
                )
        
        return recommendations
    
    def print_stats(self):
        """Print current memory statistics"""
        stats = self.get_stats()
        print("\n=== Memory Statistics ===")
        print(f"System RAM:")
        print(f"  Total: {stats.total_ram_gb:.2f} GB")
        print(f"  Available: {stats.available_ram_gb:.2f} GB")
        print(f"  Used: {stats.used_ram_gb:.2f} GB")
        print(f"\nGPU VRAM:")
        print(f"  Total: {stats.gpu_vram_gb:.2f} GB")
        print(f"  Used: {stats.used_vram_gb:.2f} GB")
        print(f"  Peak: {stats.peak_vram_gb:.2f} GB")
        print(f"\nAllocated buffers: {len(self._allocations)}")
        print("=" * 25)
