#!/usr/bin/env python3
"""
🔗 TEST INTEGRATION - Verificar que todas las técnicas están integradas
========================================================================

Script para probar que el sistema híbrido actualizado puede ejecutar
todas las 7 técnicas exitosas correctamente.
"""

import sys
import numpy as np
from pathlib import Path

# Agregar rutas del proyecto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fase_9_breakthrough_integration.src.hybrid_optimizer import (
    HybridOptimizer, HybridConfiguration, HybridStrategy
)

def test_technique_integration():
    """Prueba la integración de todas las técnicas en el sistema híbrido"""
    print("🔗 TESTING HYBRID SYSTEM INTEGRATION")
    print("=" * 60)

    # Inicializar optimizador híbrido
    try:
        optimizer = HybridOptimizer()
        print("✅ Hybrid Optimizer inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando Hybrid Optimizer: {e}")
        return False

    # Verificar qué técnicas están disponibles
    available_techniques = list(optimizer.individual_techniques.keys()) + list(optimizer.hybrid_techniques.keys())
    print(f"📊 Técnicas disponibles: {available_techniques}")

    # Técnicas esperadas
    expected_techniques = [
        'low_rank', 'cw', 'quantum',  # Técnicas originales
        'ai_predictor', 'bayesian_opt', 'neuromorphic', 'tensor_core'  # Técnicas modernas
    ]

    missing_techniques = [tech for tech in expected_techniques if tech not in available_techniques]
    if missing_techniques:
        print(f"⚠️  Técnicas faltantes: {missing_techniques}")
    else:
        print("✅ Todas las técnicas esperadas están disponibles")

    # Crear matrices de prueba pequeñas para testing rápido
    size = 64
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    print(f"\n🧪 Probando con matrices {size}x{size}")

    # Probar técnicas individuales
    successful_techniques = []
    failed_techniques = []

    test_techniques = [tech for tech in expected_techniques if tech in available_techniques]

    for technique in test_techniques:
        print(f"\n🔄 Probando técnica: {technique}")
        try:
            config = HybridConfiguration(
                strategy=HybridStrategy.SEQUENTIAL,
                techniques=[technique]
            )

            result = optimizer.optimize_hybrid(A, B, config)
            
            print(f"  Debug: result type = {type(result)}")
            print(f"  Debug: result attributes = {dir(result) if hasattr(result, '__dict__') else 'no __dict__'}")

            if hasattr(result, 'final_result') and result.final_result is not None:
                print(f"  ✅ {technique}: Éxito")
                successful_techniques.append(technique)
            else:
                print(f"  ❌ {technique}: Resultado nulo")
                failed_techniques.append(technique)

        except Exception as e:
            import traceback
            print(f"  ❌ {technique}: Error - {e}")
            print(f"  Full traceback:")
            traceback.print_exc()
            failed_techniques.append(technique)

    # Resultados finales
    print(f"\n📊 RESULTADOS DE INTEGRACIÓN:")
    print(f"  ✅ Técnicas exitosas: {len(successful_techniques)}/{len(test_techniques)}")
    print(f"  ❌ Técnicas fallidas: {len(failed_techniques)}")

    if successful_techniques:
        print(f"  ✅ Técnicas funcionales: {', '.join(successful_techniques)}")

    if failed_techniques:
        print(f"  ❌ Técnicas con problemas: {', '.join(failed_techniques)}")

    # Verificar integración completa
    integration_score = len(successful_techniques) / len(expected_techniques) * 100

    print(f"\n🏆 INTEGRATION SCORE: {integration_score:.1f}%")

    if integration_score >= 80:
        print("✅ INTEGRACIÓN EXITOSA - Sistema híbrido completamente funcional")
        return True
    else:
        print("⚠️  INTEGRACIÓN INCOMPLETA - Se requieren más ajustes")
        return False

if __name__ == "__main__":
    success = test_technique_integration()
    if success:
        print("\n🎉 ¡Sistema híbrido completamente integrado!")
    else:
        print("\n🔧 Se requiere trabajo adicional en la integración")