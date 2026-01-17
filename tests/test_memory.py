"""Tests for Memory module"""
import pytest
from core.memory import MemoryManager, MemoryStats


def test_memory_manager_initialization():
    """Test memory manager initialization"""
    manager = MemoryManager(gpu_vram_gb=8.0)
    assert manager.gpu_vram_gb == 8.0
    
    stats = manager.get_stats()
    assert isinstance(stats, MemoryStats)
    assert stats.total_ram_gb > 0
    # Check VRAM is tracked
    assert stats.used_vram_gb >= 0.0


def test_memory_stats():
    """Test memory statistics retrieval"""
    manager = MemoryManager()
    stats = manager.get_stats()
    
    assert stats.total_ram_gb > 0
    assert stats.available_ram_gb > 0
    # Check VRAM stats exist
    assert stats.used_vram_gb >= 0.0
    assert stats.available_ram_gb <= stats.total_ram_gb


def test_allocation_tracking():
    """Test memory allocation tracking"""
    manager = MemoryManager(gpu_vram_gb=8.0)
    
    # Register allocation
    success = manager.register_allocation("test_model", 2048, is_gpu=True)
    assert success
    stats = manager.get_stats()
    
    assert stats.used_vram_gb == 2.0
    
    # Free
    manager.free_allocation("test_model", is_gpu=True)
    stats = manager.get_stats()
    
    assert stats.used_vram_gb == 0.0


def test_can_allocate():
    """Test allocation feasibility check"""
    manager = MemoryManager(gpu_vram_gb=8.0)
    
    # Should be able to allocate 4GB
    can_alloc, reason = manager.can_allocate(4096, use_gpu=True)
    assert can_alloc
    
    # Should not be able to allocate more than VRAM
    can_alloc, reason = manager.can_allocate(9000, use_gpu=True)
    assert not can_alloc


def test_recommendations():
    """Test memory recommendations"""
    manager = MemoryManager(gpu_vram_gb=8.0)
    
    # Small model should fit
    recs = manager.get_recommendations(2048)
    assert recs['use_gpu']
    assert not recs['use_cpu_offload']
    
    # Large model needs optimizations
    recs = manager.get_recommendations(10000)
    assert recs['use_quantization'] or recs['use_cpu_offload']


def test_multiple_allocations():
    """Test multiple simultaneous allocations"""
    manager = MemoryManager(gpu_vram_gb=8.0)
    
    manager.register_allocation("model1", 2048, is_gpu=True)
    manager.register_allocation("model2", 1024, is_gpu=True)
    manager.register_allocation("buffer1", 512, is_gpu=True)
    
    stats = manager.get_stats()
    assert stats.used_vram_gb == 3.5
    
    # Clean up
    manager.free_allocation("model1", is_gpu=True)
    manager.free_allocation("model2", is_gpu=True)
    manager.free_allocation("buffer1", is_gpu=True)
