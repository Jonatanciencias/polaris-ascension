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
