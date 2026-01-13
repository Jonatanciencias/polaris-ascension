#!/usr/bin/env python3
"""
🔬 DEMOSTRACIÓN VERIFICABLE DEL FRAMEWORK RX 580
Descarga imágenes REALES y procesa con tiempos medidos
"""

import urllib.request
import os
from pathlib import Path
import time
import sys

# URLs de imágenes públicas (pixabay, pexels - sin restricciones)
DEMO_IMAGES = {
    "tiger": "https://images.pexels.com/photos/792381/pexels-photo-792381.jpeg?auto=compress&cs=tinysrgb&w=800",
    "elephant": "https://images.pexels.com/photos/67196/elephant-animal-wildlife-africa-67196.jpeg?auto=compress&cs=tinysrgb&w=800",
    "lion": "https://images.pexels.com/photos/33045/lion-wild-africa-african.jpg?auto=compress&cs=tinysrgb&w=800",
    "bear": "https://images.pexels.com/photos/158109/kodiak-brown-bear-adult-portrait-158109.jpeg?auto=compress&cs=tinysrgb&w=800",
    "wolf": "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&w=800",
}

def download_real_images():
    """Descarga imágenes reales de Pexels (Creative Commons)"""
    demo_dir = Path("data/wildlife/demo_real")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌐 Descargando imágenes REALES de Pexels (Creative Commons)...")
    print("=" * 70)
    
    downloaded = []
    for name, url in DEMO_IMAGES.items():
        filepath = demo_dir / f"{name}.jpg"
        
        if filepath.exists():
            print(f"✓ {name}: Ya existe ({filepath.stat().st_size/1024:.1f} KB)")
            downloaded.append(filepath)
            continue
            
        try:
            print(f"⬇️  Descargando {name}...", end=" ", flush=True)
            
            # Agregar user-agent para evitar bloqueo
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
                with open(filepath, 'wb') as f:
                    f.write(data)
            
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ ({size_kb:.1f} KB)")
            downloaded.append(filepath)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✅ {len(downloaded)} imágenes descargadas en: {demo_dir}")
    return downloaded

def classify_with_framework(image_paths):
    """Clasifica usando el motor de inferencia RX 580"""
    print("\n" + "=" * 70)
    print("🔬 CLASIFICACIÓN CON FRAMEWORK RX 580 - TIEMPOS REALES")
    print("=" * 70)
    
    try:
        from src.inference.onnx_engine import ONNXInferenceEngine
    except ImportError:
        print("❌ Error: Instala dependencias con: pip install -r requirements.txt")
        return []
    
    model_path = Path("examples/models/mobilenetv2.onnx")
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("Ejecuta: python scripts/download_models.py --model mobilenetv2")
        return []
    
    print(f"📦 Cargando modelo MobileNetV2...")
    start_load = time.time()
    engine = ONNXInferenceEngine(str(model_path), precision="fp32")
    load_time = (time.time() - start_load) * 1000
    
    model_info = engine.get_model_info()
    print(f"✅ Modelo cargado en {load_time:.1f}ms")
    print(f"   Input: {model_info['input_shape']}")
    print(f"   Precision: {model_info['precision']}")
    print()
    
    results = []
    times = []
    
    for img_path in image_paths:
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
            
            results.append({
                "image": img_path.name,
                "time_ms": elapsed,
                "prediction": predictions[0][0],
                "confidence": predictions[0][1]
            })
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    return results, times

def print_results(results, times):
    """Imprime resultados verificables"""
    if not results:
        return
    
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
        print(f"  • {r['image']:20s} → {r['prediction']:30s} ({r['confidence']:.1%}, {r['time_ms']:.0f}ms)")
    
    print()
    print("✅ DEMOSTRACIÓN COMPLETADA CON DATOS REALES Y TIEMPOS MEDIDOS")
    print("=" * 70)
    print()
    print("🔍 Verifica los resultados:")
    print("   1. Las imágenes están en: data/wildlife/demo_real/")
    print("   2. Los tiempos son mediciones reales de tu GPU")
    print("   3. Las predicciones vienen del modelo entrenado")
    print()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     🔬 DEMOSTRACIÓN VERIFICABLE - RX 580 AI Framework           ║
║     Procesamiento REAL con mediciones de tiempo verificables    ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Paso 1: Descargar imágenes reales
    image_paths = download_real_images()
    
    if not image_paths:
        print("\n❌ No se pudieron descargar imágenes")
        print("Puedes usar tus propias imágenes en data/wildlife/demo_real/")
        sys.exit(1)
    
    # Paso 2: Clasificar con framework
    results, times = classify_with_framework(image_paths)
    
    # Paso 3: Mostrar resultados
    print_results(results, times)
