"""Tests for GPU module"""
import pytest
from core.gpu import GPUManager, GPUInfo


def test_gpu_manager_initialization():
    """Test GPU manager can be initialized"""
    manager = GPUManager()
    assert manager is not None
    assert not manager.is_initialized()


def test_gpu_detection():
    """Test GPU detection"""
    manager = GPUManager()
    gpu_info = manager.detect_gpu()
    
    # Should detect GPU or return None
    if gpu_info:
        assert isinstance(gpu_info, GPUInfo)
        assert gpu_info.name
        assert gpu_info.pci_id
        assert gpu_info.architecture
        print(f"\nDetected: {gpu_info.name}")
    else:
        print("\nNo GPU detected (may be expected in CI)")


def test_gpu_initialization():
    """Test GPU initialization"""
    manager = GPUManager()
    result = manager.initialize()
    
    # In CI without GPU, this may fail
    if result:
        assert manager.is_initialized()
        assert manager.get_info() is not None


def test_compute_backend_detection():
    """Test compute backend detection"""
    manager = GPUManager()
    manager.detect_gpu()
    
    backend = manager.get_compute_backend()
    assert backend in ['rocm', 'opencl', 'cpu']
    print(f"\nCompute backend: {backend}")


def test_gpu_info_dataclass():
    """Test GPUInfo dataclass"""
    info = GPUInfo(
        name="Test GPU",
        pci_id="00:00.0",
        architecture="Test Arch",
        vram_mb=8192
    )
    
    assert info.name == "Test GPU"
    assert info.pci_id == "00:00.0"
    assert info.vram_mb == 8192
    assert not info.opencl_available
    assert not info.rocm_available
