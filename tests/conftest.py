"""
Test Configuration
==================

Configuración global para pytest.
Los tests en el directorio 'legacy/' están excluidos automáticamente.
"""
import sys
import os
import pytest

# Add project root and src to path for testing
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

GPU_OPENCL_TEST_FILES = {
    "test_opencl_gemm.py",
    "test_optimized_kernel_engine.py",
    "test_system_integration.py",
    "test_advanced_memory_manager.py",
}

INTEGRATION_TEST_FILES = {
    "test_system_integration.py",
}


def pytest_ignore_collect(collection_path, config):
    """Ignorar el directorio legacy durante la recolección de tests"""
    if 'legacy' in str(collection_path):
        return True
    return False


def pytest_configure(config):
    """Configuración adicional de pytest"""
    # Registrar marcadores personalizados
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "opencl: marks tests that require OpenCL runtime/device"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Clasificación centralizada por marcadores para estabilidad de ejecución."""
    for item in items:
        file_name = os.path.basename(str(item.fspath))
        node_id_lc = item.nodeid.lower()

        if file_name in GPU_OPENCL_TEST_FILES:
            item.add_marker(pytest.mark.gpu)
            item.add_marker(pytest.mark.opencl)

        if file_name in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        if "benchmark" in node_id_lc and "slow" not in item.keywords:
            item.add_marker(pytest.mark.slow)
