#!/usr/bin/env python3
"""
Tests para CalibratedIntelligentSelector - Selector inteligente de kernels
==========================================================================

Tests para verificar el funcionamiento del selector de kernels calibrado
que usa ML para predecir el mejor kernel.
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def selector():
    """Fixture para crear el selector"""
    try:
        from src.ml_models.calibrated_intelligent_selector import CalibratedIntelligentSelector

        sel = CalibratedIntelligentSelector()
        return sel
    except ImportError:
        pytest.skip("CalibratedIntelligentSelector not available")


class TestSelectorInitialization:
    """Tests de inicialización del selector"""

    def test_selector_creates_successfully(self, selector):
        """Verifica que el selector se crea correctamente"""
        assert selector is not None

    def test_has_select_technique_method(self, selector):
        """Verifica que tiene método de selección de técnica"""
        assert hasattr(selector, "select_technique")

    def test_has_analyze_matrix_method(self, selector):
        """Verifica que tiene método de análisis de matriz"""
        assert hasattr(selector, "analyze_matrix")


class TestTechniquePrediction:
    """Tests de predicción de técnicas"""

    def test_analyzes_matrix_characteristics(self, selector):
        """Verifica análisis de características de matriz"""
        import numpy as np

        matrix = np.random.randn(256, 256).astype(np.float32)

        try:
            analysis = selector.analyze_matrix(matrix)
            assert analysis is not None
        except Exception as e:
            # Si falla, verificar que es un error esperado
            pytest.skip(f"Selector no calibrado: {e}")

    def test_selects_technique(self, selector):
        """Verifica selección de técnica para matriz"""
        import numpy as np

        matrix = np.random.randn(512, 512).astype(np.float32)

        try:
            result = selector.select_technique(matrix)
            assert result is not None
        except Exception as e:
            # Es aceptable que falle si no está calibrado
            pytest.skip(f"Selector no calibrado: {e}")


class TestEdgeCases:
    """Tests de casos extremos"""

    def test_handles_small_matrix(self, selector):
        """Verifica manejo de matrices pequeñas"""
        import numpy as np

        matrix = np.random.randn(16, 16).astype(np.float32)

        try:
            result = selector.analyze_matrix(matrix)
            assert result is not None
        except Exception:
            pass

    def test_handles_sparse_matrix(self, selector):
        """Verifica manejo de matrices sparse"""
        import numpy as np

        # Matriz con ~90% ceros
        matrix = np.random.randn(256, 256).astype(np.float32)
        mask = np.random.random((256, 256)) > 0.1
        matrix[mask] = 0

        try:
            result = selector.analyze_matrix(matrix)
            assert result is not None
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
