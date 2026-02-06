#!/bin/bash

# Script de inicio para Fase 1 del Roadmap de Optimización
# Este script prepara el entorno y marca el inicio de la primera fase

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "   🚀 INICIO FASE 1: QUICK WINS"
echo "=========================================="
echo ""

# Verificar que estamos en el directorio correcto
cd "$PROJECT_ROOT"

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "✅ Activando entorno virtual..."
    source venv/bin/activate
else
    echo "⚠️  Entorno virtual no encontrado. Asegúrate de ejecutar 'python -m venv venv' primero."
fi

# Verificar que el script de progreso existe
if [ ! -f "scripts/update_progress.py" ]; then
    echo "❌ Error: scripts/update_progress.py no encontrado"
    exit 1
fi

echo ""
echo "📊 Estado actual del proyecto:"
echo "----------------------------------------"
python scripts/update_progress.py --summary
echo ""

# Confirmar inicio de fase
echo "¿Deseas iniciar la Fase 1 del roadmap? (s/n)"
read -r response

if [[ "$response" =~ ^[Ss]$ ]]; then
    echo ""
    echo "📝 Marcando inicio de Fase 1..."
    
    # Crear entrada en el log (actualización manual del PROGRESS_TRACKING.md)
    echo ""
    echo "⚡ Tareas de Fase 1:"
    echo "   1.1 Corrección de Kernels (Tasks 1.1.1-1.1.4)"
    echo "   1.2 Optimización de GCN4_VEC4 (Tasks 1.2.1-1.2.3)"
    echo "   1.3 Tuning de Hiperparámetros (Tasks 1.3.1-1.3.6)"
    echo ""
    
    echo "🎯 Objetivo Fase 1: 180-200 GFLOPS"
    echo "📅 Duración estimada: 1-2 semanas"
    echo "📍 Baseline actual: 150.96 GFLOPS"
    echo ""
    
    echo "💡 Primera tarea sugerida: Task 1.1.1 - Diagnosticar error FLOAT4"
    echo ""
    echo "Para iniciar esta tarea, ejecuta:"
    echo "   python scripts/update_progress.py --task 1.1.1 --status in-progress"
    echo ""
    
    # Verificar tests actuales
    echo "🧪 Verificando tests antes de comenzar..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || echo "⚠️  Algunos tests fallaron. Revisa antes de continuar."
    else
        echo "⚠️  pytest no encontrado. Instala con: pip install pytest"
    fi
    
    echo ""
    echo "✅ Sistema listo para Fase 1"
    echo ""
    echo "📖 Recursos útiles:"
    echo "   - Roadmap completo: docs/ROADMAP_OPTIMIZATION.md"
    echo "   - Tracking diario: docs/PROGRESS_TRACKING.md"
    echo "   - Guía del sistema: docs/ROADMAP_README.md"
    echo "   - Benchmark baseline: results/hardware_benchmark_rx590_gme.md"
    echo ""
    echo "🔗 Comandos rápidos:"
    echo "   - Ver estado: python scripts/update_progress.py --summary"
    echo "   - Iniciar tarea: python scripts/update_progress.py --task X.Y.Z --status in-progress"
    echo "   - Completar tarea: python scripts/update_progress.py --task X.Y.Z --status completed"
    echo "   - Registrar GFLOPS: python scripts/update_progress.py --gflops XXX.XX --notes 'descripción'"
    echo ""
    echo "¡Buena suerte! 🚀"
    
else
    echo ""
    echo "❌ Inicio cancelado. Ejecuta este script nuevamente cuando estés listo."
fi

echo ""
echo "=========================================="
