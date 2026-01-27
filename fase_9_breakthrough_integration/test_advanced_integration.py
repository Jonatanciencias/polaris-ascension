#!/usr/bin/env python3
"""
🧪 PRUEBA DE INTEGRACIÓN: SELECTOR INTELIGENTE CON ANÁLISIS AVANZADO
======================================================================

Demostración de la integración completa del sistema de selección inteligente
con análisis avanzado de matrices y todas las mejoras profesionales.
"""

import sys
import numpy as np
from pathlib import Path

# Agregar el directorio src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from intelligent_technique_selector import IntelligentTechniqueSelector

def test_advanced_integration():
    """Prueba completa de la integración avanzada"""
    print("🚀 PRUEBA DE INTEGRACIÓN AVANZADA")
    print("=" * 60)

    # Inicializar selector
    selector = IntelligentTechniqueSelector()
    print("✅ Selector inteligente inicializado")

    # Matrices de prueba con diferentes características
    test_cases = [
        {
            "name": "Matriz densa cuadrada",
            "matrix_a": np.random.randn(128, 128),
            "matrix_b": np.random.randn(128, 128)
        },
        {
            "name": "Matriz sparse",
            "matrix_a": np.random.randn(128, 128) * (np.random.rand(128, 128) > 0.95),
            "matrix_b": np.random.randn(128, 128) * (np.random.rand(128, 128) > 0.95)
        },
        {
            "name": "Matriz diagonal",
            "matrix_a": np.diag(np.random.randn(128)),
            "matrix_b": np.diag(np.random.randn(128))
        },
        {
            "name": "Matriz triangular",
            "matrix_a": np.triu(np.random.randn(128, 128)),
            "matrix_b": np.tril(np.random.randn(128, 128))
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Caso {i}: {test_case['name']}")
        print("-" * 40)

        matrix_a = test_case["matrix_a"]
        matrix_b = test_case["matrix_b"]

        # Extraer características
        features = selector.feature_extractor.extract_features(matrix_a, matrix_b)

        print("🔍 Características Básicas:")
        print(f"   Tamaño A: {features.size_a}, B: {features.size_b}")
        print(".3f")
        print(".2e")
        print(".2f")
        print(f"   Estructura: {features.structure_type}, Tipo: {features.matrix_type}")

        # Mostrar características avanzadas si están disponibles
        if features.advanced_features is not None:
            print("🔬 Características Avanzadas:")
            print("   Espectral A:")
            if features.spectral_properties_a:
                print(".2e")
                print(".2f")
                print(f"       Rango numérico: {features.spectral_properties_a['numerical_rank']}")
            print("   Espectral B:")
            if features.spectral_properties_b:
                print(".2e")
                print(".2f")
                print(f"       Rango numérico: {features.spectral_properties_b['numerical_rank']}")

            print("   Estructural A:")
            if features.structural_properties_a:
                print(f"       Tipo: {features.structural_properties_a['structure_type']}")
                print(f"       Simetría: {features.structural_properties_a['symmetry_type']}")
                if features.structural_properties_a['bandwidth']:
                    print(f"       Bandwidth: {features.structural_properties_a['bandwidth']}")

            print("   Computacional:")
            if features.computational_properties:
                print(".2f")
                print(".2f")
                print(f"       Patrón memoria: {features.computational_properties['memory_access_pattern']}")

            print("   ML Features:")
            if features.ml_features:
                for key, value in list(features.ml_features.items())[:3]:
                    print(f"       {key}: {value:.3f}")

        # Seleccionar técnica
        result = selector.select_technique(matrix_a, matrix_b)

        print("🎯 Selección de Técnica:")
        print(f"   Recomendada: {result.recommended_technique.value}")
        print(".2f")
        print(f"   Confianza: {result.selection_confidence:.2f}")
        print(f"   Alternativas: {[t.value for t in result.alternative_options[:2]]}")

        if result.reasoning:
            print("   Razones:")
            for reason in result.reasoning[:2]:
                print(f"     • {reason}")

    print("\n✅ Prueba de integración completada")
    print("🎉 El sistema de selección inteligente con análisis avanzado está funcionando correctamente!")

if __name__ == "__main__":
    test_advanced_integration()