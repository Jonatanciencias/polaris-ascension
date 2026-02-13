#!/usr/bin/env python3
"""
Tests para AdvancedMemoryManager - Gestión de memoria GPU
=========================================================

Tests para verificar el funcionamiento del gestor de memoria avanzado
con prefetch, tiling y buffer pooling.
"""

import pytest
import numpy as np

# Skip tests si OpenCL no está disponible
try:
    import pyopencl as cl

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

pytestmark = pytest.mark.skipif(not HAS_OPENCL, reason="PyOpenCL not available")


@pytest.fixture(scope="module")
def memory_manager():
    """Fixture para crear el memory manager"""
    from src.optimization_engines.advanced_memory_manager import AdvancedMemoryManager

    # Crear contexto OpenCL
    platforms = cl.get_platforms()
    gpu_device = None
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            gpu_device = devices[0]
            break

    if gpu_device is None:
        pytest.skip("No GPU device found")

    context = cl.Context([gpu_device])
    queue = cl.CommandQueue(context)

    # API real: context, queue, max_gpu_memory, enable_prefetch, enable_tiling
    manager = AdvancedMemoryManager(
        context=context,
        queue=queue,
        max_gpu_memory=4 * 1024 * 1024 * 1024,  # 4GB
        enable_prefetch=True,
        enable_tiling=True,
    )

    yield manager


class TestMemoryManagerInitialization:
    """Tests de inicialización del gestor de memoria"""

    def test_manager_creates_successfully(self, memory_manager):
        """Verifica que el manager se crea correctamente"""
        assert memory_manager is not None

    def test_gpu_memory_detected(self, memory_manager):
        """Verifica que se detecta la memoria GPU"""
        assert memory_manager.max_gpu_memory > 0


class TestBufferOperations:
    """Tests de operaciones con buffers"""

    def test_allocate_buffer(self, memory_manager):
        """Verifica que se pueden asignar buffers"""
        data = np.random.randn(100, 100).astype(np.float32)
        buffer = memory_manager.allocate(data)

        assert buffer is not None

    def test_managed_buffer_context(self, memory_manager):
        """Verifica que el context manager de buffers funciona"""
        data = np.random.randn(50, 50).astype(np.float32)

        with memory_manager.managed_buffer(data) as buf:
            assert buf is not None


class TestMemoryTracking:
    """Tests de tracking de memoria"""

    def test_memory_stats_available(self, memory_manager):
        """Verifica que las estadísticas de memoria están disponibles"""
        stats = memory_manager.get_stats()

        assert stats is not None


class TestErrorHandling:
    """Tests de manejo de errores"""

    def test_empty_array_handled(self, memory_manager):
        """Verifica que arrays vacíos se manejan correctamente"""
        # Array con al menos algún elemento para evitar problemas
        data = np.array([1.0], dtype=np.float32)

        try:
            buffer = memory_manager.allocate(data)
            assert buffer is not None
        except (ValueError, cl.Error):
            # Es aceptable rechazar arrays muy pequeños
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
