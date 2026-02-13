#!/usr/bin/env python3
"""
🔍 ANÁLISIS RÁPIDO DEL SISTEMA DE SELECCIÓN INTELIGENTE
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Agregar path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "fase_9_breakthrough_integration" / "src"))

from intelligent_technique_selector import IntelligentTechniqueSelector


def quick_analysis():
    print("🚀 ANÁLISIS RÁPIDO DEL SISTEMA DE SELECCIÓN INTELIGENTE")
    print("=" * 60)

    selector = IntelligentTechniqueSelector()

    # Matrices de prueba
    test_cases = [
        ("Pequeña densa", np.random.randn(256, 256), np.random.randn(256, 256)),
        ("Mediana densa", np.random.randn(512, 512), np.random.randn(512, 512)),
        (
            "Grande sparse",
            np.random.randn(512, 512) * (np.random.rand(512, 512) > 0.8),
            np.random.randn(512, 512) * (np.random.rand(512, 512) > 0.8),
        ),
    ]

    results = []
    techniques_used = {}

    for name, A, B in test_cases:
        print(f"\n📊 {name} ({A.shape}):")

        start = time.time()
        result = selector.select_technique(A, B)
        elapsed = time.time() - start

        tech = result.recommended_technique.value
        techniques_used[tech] = techniques_used.get(tech, 0) + 1

        print(f"   Recomendado: {tech}")
        print(f"   Confianza: {result.selection_confidence:.2f}")
        print(f"   Performance: {result.expected_performance:.1f} GFLOPS")
        print(f"   Tiempo: {elapsed:.3f}s")

        results.append(
            {
                "case": name,
                "technique": tech,
                "confidence": result.selection_confidence,
                "performance": result.expected_performance,
                "time": elapsed,
            }
        )

    print("\n📈 RESUMEN:")
    print(f"   Técnicas distintas usadas: {len(techniques_used)}")
    print(f"   Confianza media: {np.mean([r['confidence'] for r in results]):.2f}")
    print(f"   Tiempo medio: {np.mean([r['time'] for r in results]):.3f}s")

    # Evaluar si vale la pena mejorar
    print("\n🎯 EVALUACIÓN DE MEJORAS PROPUESTAS:")
    print("=" * 60)

    improvements = {
        "Calibración de pesos": "🔴 ALTA - Sistema usa pesos fijos, necesita optimización automática",
        "Dataset expandido": "🔴 ALTA - No existe dataset de entrenamiento",
        "Combinaciones automáticas": "🟡 MEDIA - Podría mejorar performance significativamente",
        "Métricas avanzadas": "🟡 MEDIA - Sistema básico funciona, pero se puede enriquecer",
    }

    for improvement, evaluation in improvements.items():
        print(f"   {improvement}:")
        print(f"      {evaluation}")

    print("\n✅ CONCLUSIÓN: Las mejoras valen la pena y se implementarán profesionalmente")

    # Guardar resultados
    output_file = project_root / "fase_9_breakthrough_integration" / "data" / "quick_analysis.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "results": results,
                "techniques_used": techniques_used,
                "recommendations": list(improvements.keys()),
            },
            f,
            indent=2,
        )

    print(f"\n💾 Resultados guardados en: {output_file}")


if __name__ == "__main__":
    quick_analysis()
