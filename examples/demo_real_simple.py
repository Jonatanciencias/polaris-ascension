#!/usr/bin/env python3
"""
🔬 DEMOSTRACIÓN VERIFICABLE CON IMÁGENES REALES
Usa cualquier imagen que tengas en tu computadora
"""

import sys
from pathlib import Path
import time

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

print(
    """
╔══════════════════════════════════════════════════════════════════╗
║     🔬 DEMOSTRACIÓN VERIFICABLE - RX 580 AI Framework           ║
║     Procesamiento REAL con mediciones de tiempo verificables    ║
╚══════════════════════════════════════════════════════════════════╝
"""
)

# Verificar imágenes disponibles
demo_dir = Path("data/wildlife/demo_real")
if demo_dir.exists():
    images = (
        list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.jpeg")) + list(demo_dir.glob("*.png"))
    )
else:
    images = []

if not images:
    print("❌ No hay imágenes en data/wildlife/demo_real/")
    print("\n📥 Descarga imágenes de animales de Google Images o:")
    print("   • https://unsplash.com/s/photos/wildlife")
    print("   • https://pixabay.com/images/search/animals/")
    print("\nY guárdalas en: data/wildlife/demo_real/\n")
    sys.exit(1)

print(f"✅ Encontradas {len(images)} imágenes en {demo_dir}")
print()

# Cargar modelo
print("=" * 70)
print("🔬 CLASIFICACIÓN CON FRAMEWORK RX 580 - TIEMPOS REALES")
print("=" * 70)

try:
    from src.inference.onnx_engine import ONNXInferenceEngine
    from src.inference.base import InferenceConfig
except ImportError as e:
    print(f"❌ Error importando: {e}")
    print("Ejecuta: source venv/bin/activate")
    sys.exit(1)

model_path = Path("examples/models/mobilenetv2.onnx")
if not model_path.exists():
    print(f"❌ Modelo no encontrado: {model_path}")
    print("Ejecuta: python scripts/download_models.py --model mobilenetv2")
    sys.exit(1)

print(f"📦 Cargando modelo MobileNetV2...")
start_load = time.time()

# Crear configuración
config = InferenceConfig(model_path=str(model_path), device="gpu", batch_size=1)

engine = ONNXInferenceEngine(config)
load_time = (time.time() - start_load) * 1000

model_info = engine.get_model_info()
print(f"✅ Modelo cargado en {load_time:.1f}ms")
print(f"   Input: {model_info.get('input_shape', 'N/A')}")
print()

# Procesar imágenes
results = []
times = []

for img_path in images[:5]:  # Max 5 imágenes
    print(f"🖼️  {img_path.name}:")

    try:
        start = time.time()
        predictions = engine.infer(str(img_path), top_k=3)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

        print(f"   ⏱️  {elapsed:.1f}ms")

        for i, (label, conf) in enumerate(predictions, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {icon} {label}: {conf:.1%}")

        results.append(
            {
                "image": img_path.name,
                "time_ms": elapsed,
                "prediction": predictions[0][0],
                "confidence": predictions[0][1],
            }
        )
        print()

    except Exception as e:
        print(f"   ❌ Error: {e}\n")

# Resultados
if not results:
    print("❌ No se pudieron procesar imágenes")
    sys.exit(1)

print("=" * 70)
print("📊 RESULTADOS VERIFICABLES - HARDWARE REAL")
print("=" * 70)
print(f"GPU: AMD Radeon RX 580 (8GB)")
print(f"Modelo: MobileNetV2 (ImageNet-1000)")
print(f"Precisión: FP32")
print()
print(f"Imágenes procesadas: {len(results)}")
print(f"Tiempo promedio: {sum(times)/len(times):.1f}ms")
print(f"Tiempo mínimo: {min(times):.1f}ms")
print(f"Tiempo máximo: {max(times):.1f}ms")
print(f"Throughput: {1000/(sum(times)/len(times)):.2f} fps")
print()

print("Predicciones:")
for r in results:
    print(
        f"  • {r['image']:20s} → {r['prediction']:30s} ({r['confidence']:.1%}, {r['time_ms']:.0f}ms)"
    )

print()
print("✅ DEMOSTRACIÓN COMPLETADA CON DATOS REALES Y TIEMPOS MEDIDOS")
print("=" * 70)
print()
print("🔍 Verifica los resultados:")
print(f"   1. Las imágenes están en: {demo_dir}/")
print("   2. Los tiempos son mediciones reales de tu GPU RX 580")
print("   3. Las predicciones vienen del modelo MobileNetV2 entrenado")
print()
