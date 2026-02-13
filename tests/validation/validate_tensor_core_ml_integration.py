#!/usr/bin/env python3
"""
🎯 VALIDACIÓN DE INTEGRACIÓN TENSOR CORE CON ML PREDICTOR
==========================================================

Script para validar que Tensor Core Simulation está correctamente integrado
con el sistema ML-based de selección de técnicas.

FASE 10.1: Validación de integración Tensor Core + ML
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Añadir paths necesarios
project_root = Path(__file__).parent
sys.path.append(str(project_root / "fase_9_breakthrough_integration" / "src"))

try:
    from breakthrough_selector import BreakthroughTechniqueSelector, BreakthroughTechnique

    SELECTOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importando Breakthrough Selector: {e}")
    SELECTOR_AVAILABLE = False


class TensorCoreMLIntegrationValidator:
    """
    Validador de la integración de Tensor Core con el sistema ML.
    """

    def __init__(self):
        self.selector = None
        self.test_cases = []
        self.results = []

    def initialize_selector(self) -> bool:
        """Inicializa el Breakthrough Selector con Tensor Core."""
        if not SELECTOR_AVAILABLE:
            print("❌ Breakthrough Selector no disponible")
            return False

        try:
            self.selector = BreakthroughTechniqueSelector(
                use_ml_predictor=True, use_bayesian_opt=False
            )
            print("✅ Breakthrough Selector inicializado con Tensor Core")
            return True
        except Exception as e:
            print(f"❌ Error inicializando selector: {e}")
            return False

    def create_test_cases(self):
        """Crea casos de prueba para validar selección de Tensor Core."""
        self.test_cases = [
            {
                "name": "Matriz Cuadrada Pequeña (128x128)",
                "matrices": (
                    np.random.randn(128, 128).astype(np.float32),
                    np.random.randn(128, 128).astype(np.float32),
                ),
                "expected_technique": BreakthroughTechnique.TRADITIONAL,
                "reason": "Demasiado pequeña para Tensor Core",
            },
            {
                "name": "Matriz Cuadrada Mediana (256x256)",
                "matrices": (
                    np.random.randn(256, 256).astype(np.float32),
                    np.random.randn(256, 256).astype(np.float32),
                ),
                "expected_technique": BreakthroughTechnique.TENSOR_CORE_SIMULATION,
                "reason": "Tamaño ideal para Tensor Core",
            },
            {
                "name": "Matriz Cuadrada Grande (512x512)",
                "matrices": (
                    np.random.randn(512, 512).astype(np.float32),
                    np.random.randn(512, 512).astype(np.float32),
                ),
                "expected_technique": BreakthroughTechnique.TENSOR_CORE_SIMULATION,
                "reason": "Tamaño óptimo para Tensor Core",
            },
            {
                "name": "Matriz Rectangular (256x512 x 512x256)",
                "matrices": (
                    np.random.randn(256, 512).astype(np.float32),
                    np.random.randn(512, 256).astype(np.float32),
                ),
                "expected_technique": BreakthroughTechnique.COPPERSMITH_WINOGRAD,
                "reason": "No cuadrada, mejor CW",
            },
            {
                "name": "Matriz Sparsa (256x256, 90% zeros)",
                "matrices": self._create_sparse_matrix(256, 0.9),
                "expected_technique": BreakthroughTechnique.LOW_RANK,
                "reason": "Muy sparsa, mejor Low-Rank",
            },
        ]

    def _create_sparse_matrix(self, size: int, sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Crea matrices dispersas para testing."""
        # Crear matriz con muchos ceros
        a = np.random.randn(size, size).astype(np.float32)
        mask_a = np.random.random((size, size)) < sparsity
        a[mask_a] = 0.0

        b = np.random.randn(size, size).astype(np.float32)
        mask_b = np.random.random((size, size)) < sparsity
        b[mask_b] = 0.0

        return a, b

    def run_validation(self) -> bool:
        """Ejecuta la validación completa."""
        print("🧠 TENSOR CORE ML INTEGRATION VALIDATION")
        print("=" * 60)

        if not self.initialize_selector():
            return False

        self.create_test_cases()

        success_count = 0
        total_cases = len(self.test_cases)

        for i, test_case in enumerate(self.test_cases):
            print(f"\n🔬 Test Case {i+1}/{total_cases}: {test_case['name']}")
            print(f"   Expected: {test_case['expected_technique'].value}")
            print(f"   Reason: {test_case['reason']}")

            try:
                # Ejecutar selección
                matrix_a, matrix_b = test_case["matrices"]
                selection = self.selector.select_technique(matrix_a, matrix_b)

                print(f"   Selected: {selection.technique.value}")
                print(f"   Expected Performance: {selection.expected_performance:.2f} GFLOPS")
                print(f"   Confidence: {selection.confidence:.2f}")

                # Verificar si la selección es correcta
                if selection.technique == test_case["expected_technique"]:
                    print("   ✅ CORRECTO")
                    success_count += 1
                else:
                    print("   ❌ INCORRECTO")
                    print(f"      Expected: {test_case['expected_technique'].value}")

                # Ejecutar la técnica seleccionada
                print("   🚀 Ejecutando técnica seleccionada...")
                result, metrics = self.selector.execute_selected_technique(
                    matrix_a, matrix_b, selection
                )

                print(f"   Actual GFLOPS: {metrics.get('gflops_achieved', 0):.2f}")
                print(f"   Technique: {metrics.get('technique', 'unknown')}")

                # Validar resultado
                expected_result = np.dot(matrix_a, matrix_b)
                max_error = np.max(np.abs(result - expected_result))

                if max_error < 1e-2:
                    accuracy_status = "✅ EXCELENTE"
                elif max_error < 1e-1:
                    accuracy_status = "⚠️ BUENA"
                elif max_error < 1.0:
                    accuracy_status = "⚠️ ACEPTABLE"
                else:
                    accuracy_status = "❌ REQUIERE DEBUG"

                print(f"   Max Error: {max_error:.2e}")
                print(f"   Accuracy: {accuracy_status}")

                # Registrar resultado
                self.results.append(
                    {
                        "test_case": test_case["name"],
                        "expected": test_case["expected_technique"].value,
                        "selected": selection.technique.value,
                        "correct": selection.technique == test_case["expected_technique"],
                        "confidence": selection.confidence,
                        "performance": selection.expected_performance,
                        "actual_gflops": metrics.get("gflops_achieved", 0),
                        "max_error": max_error,
                        "accuracy_status": accuracy_status,
                    }
                )

            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.results.append(
                    {
                        "test_case": test_case["name"],
                        "expected": test_case["expected_technique"].value,
                        "selected": "ERROR",
                        "correct": False,
                        "confidence": 0.0,
                        "performance": 0.0,
                        "actual_gflops": 0.0,
                        "max_error": float("inf"),
                        "accuracy_status": "ERROR",
                    }
                )

        # Resumen final
        print("\n📊 RESUMEN DE VALIDACIÓN")
        print("-" * 40)
        print(f"Casos de prueba: {total_cases}")
        print(f"Selecciones correctas: {success_count}")
        success_rate = success_count / total_cases
        print(f"Tasa de éxito: {success_rate:.1f}")

        if success_rate >= 0.8:
            print("🎉 INTEGRACIÓN EXITOSA")
            return True
        else:
            print("⚠️ INTEGRACIÓN REQUIERE AJUSTES")
            return False


if __name__ == "__main__":
    validator = TensorCoreMLIntegrationValidator()
    success = validator.run_validation()

    if success:
        print("\n✅ Tensor Core ML Integration validation completed successfully!")
    else:
        print("\n❌ Tensor Core ML Integration validation failed!")
        sys.exit(1)
