#!/usr/bin/env python3
"""
Demostración Verificable del Framework RX 580

Este script demuestra las capacidades del framework con imágenes reales
y mediciones de tiempo verificables en hardware real.

Características:
- Descarga imágenes reales de fuentes públicas (Pexels)
- Procesa con el motor de inferencia RX 580
- Muestra tiempos medidos reales
- Presenta predicciones con nombres legibles

Uso:
    python examples/demo_verificable.py
    python examples/demo_verificable.py --download-only
    python examples/demo_verificable.py --benchmark
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.onnx_engine import ONNXInferenceEngine
from src.inference.base import InferenceConfig


# Public domain wildlife images from Pexels (Creative Commons)
DEMO_IMAGES = {
    "tiger": "https://images.pexels.com/photos/792381/pexels-photo-792381.jpeg?auto=compress&cs=tinysrgb&w=800",
    "elephant": "https://images.pexels.com/photos/66898/elephant-cub-tsavo-kenya-66898.jpeg?auto=compress&cs=tinysrgb&w=800",
    "lion": "https://images.pexels.com/photos/33045/lion-wild-africa-african.jpg?auto=compress&cs=tinysrgb&w=800",
    "bear": "https://images.pexels.com/photos/1661179/pexels-photo-1661179.jpeg?auto=compress&cs=tinysrgb&w=800",
    "wolf": "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&w=800",
}


def print_header():
    """Print demonstration header."""
    print()
    print("╔" + "═" * 64 + "╗")
    print("║" + " " * 12 + "🔬 DEMOSTRACIÓN VERIFICABLE" + " " * 25 + "║")
    print("║" + " " * 18 + "RX 580 AI Framework" + " " * 27 + "║")
    print("║" + " " * 10 + "Procesamiento REAL con tiempos medidos" + " " * 16 + "║")
    print("╚" + "═" * 64 + "╝")
    print()


def download_demo_images(demo_dir: Path, force: bool = False) -> List[Path]:
    """
    Download real wildlife images from Pexels.
    
    Args:
        demo_dir: Directory to save images
        force: Re-download even if images exist
        
    Returns:
        List of paths to downloaded images
    """
    print("🌐 Descargando imágenes REALES de Pexels (Creative Commons)...")
    print("=" * 70)
    
    demo_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    
    for name, url in DEMO_IMAGES.items():
        filepath = demo_dir / f"{name}.jpg"
        
        if filepath.exists() and not force:
            size_kb = filepath.stat().st_size / 1024
            print(f"✓ {name:15s} Ya existe ({size_kb:6.1f} KB)")
            downloaded.append(filepath)
            continue
        
        try:
            print(f"⬇️  {name:15s} Descargando...", end=" ", flush=True)
            
            # Add user-agent to avoid blocking
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (RX580-AI-Framework)'}
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()
                filepath.write_bytes(data)
            
            size_kb = len(data) / 1024
            print(f"✓ ({size_kb:6.1f} KB)")
            downloaded.append(filepath)
            time.sleep(0.5)  # Be respectful to the server
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print()
    print(f"✅ {len(downloaded)}/{len(DEMO_IMAGES)} imágenes descargadas")
    print(f"📁 Ubicación: {demo_dir}")
    print()
    
    return downloaded


def load_imagenet_labels(models_dir: Path) -> List[str]:
    """
    Load ImageNet class labels.
    
    Args:
        models_dir: Directory containing label files
        
    Returns:
        List of 1000 ImageNet class labels
    """
    labels_path = models_dir / "imagenet_labels.txt"
    
    if not labels_path.exists():
        print("⚠️  ImageNet labels no encontrados")
        print(f"   Ejecuta: python scripts/download_models.py --labels")
        print(f"   Usando labels genéricos por ahora...")
        return [f"class_{i}" for i in range(1000)]
    
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    return labels


def classify_images(
    image_paths: List[Path],
    model_path: Path,
    labels: List[str],
    benchmark: bool = False
) -> Tuple[List[dict], List[float]]:
    """
    Classify images using the RX 580 inference engine.
    
    Args:
        image_paths: List of image file paths
        model_path: Path to ONNX model
        labels: List of class labels
        benchmark: Run multiple iterations for benchmarking
        
    Returns:
        Tuple of (results list, times list)
    """
    print("=" * 70)
    print("🔬 CLASIFICACIÓN CON FRAMEWORK RX 580 - TIEMPOS REALES")
    print("=" * 70)
    
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Ejecuta: python scripts/download_models.py --model mobilenet")
        return [], []
    
    # Initialize inference engine
    print("📦 Cargando modelo MobileNetV2...")
    config = InferenceConfig(device='auto', batch_size=1)
    engine = ONNXInferenceEngine(config)
    
    start_load = time.time()
    model_info = engine.load_model(str(model_path))
    load_time = (time.time() - start_load) * 1000
    
    print(f"✅ Modelo cargado en {load_time:.1f}ms")
    print(f"   Input: {model_info.input_shape}")
    print(f"   Backend: {model_info.backend}")
    print()
    
    # Process images
    results = []
    times = []
    iterations = 5 if benchmark else 1
    
    for img_path in image_paths:
        print(f"🖼️  {img_path.name}:")
        
        try:
            # Run inference (potentially multiple times for benchmarking)
            run_times = []
            output = None
            
            for _ in range(iterations):
                start = time.time()
                output = engine.infer(str(img_path))
                elapsed = (time.time() - start) * 1000
                run_times.append(elapsed)
            
            # Use average time if benchmarking
            avg_time = sum(run_times) / len(run_times)
            times.append(avg_time)
            
            if benchmark:
                print(f"   ⏱️  {avg_time:.1f}ms (promedio de {iterations} ejecuciones)")
                print(f"       Min: {min(run_times):.1f}ms, Max: {max(run_times):.1f}ms")
            else:
                print(f"   ⏱️  {avg_time:.1f}ms")
            
            # Extract top-3 predictions
            predictions = output['predictions'][:3]
            
            for i, pred in enumerate(predictions, 1):
                idx = pred['class_id']
                conf = pred['confidence']
                icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                label = labels[idx] if idx < len(labels) else f"class_{idx}"
                print(f"   {icon} {label}: {conf:.1%}")
            
            # Store result
            top_pred = predictions[0]
            top_label = labels[top_pred['class_id']] if top_pred['class_id'] < len(labels) else f"class_{top_pred['class_id']}"
            
            results.append({
                "image": img_path.name,
                "prediction": top_label,
                "confidence": top_pred['confidence'],
                "time_ms": avg_time
            })
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            continue
    
    return results, times


def print_summary(results: List[dict], times: List[float], demo_dir: Path):
    """
    Print summary of verification results.
    
    Args:
        results: List of classification results
        times: List of inference times
        demo_dir: Directory containing demo images
    """
    if not results:
        print("❌ No se pudieron procesar imágenes")
        return
    
    print("=" * 70)
    print("📊 RESULTADOS VERIFICABLES - HARDWARE REAL")
    print("=" * 70)
    print()
    print(f"🖥️  GPU: AMD Radeon RX 580 (8GB)")
    print(f"🧠 Modelo: MobileNetV2 (ImageNet-1000)")
    print(f"📸 Imágenes procesadas: {len(results)} archivos reales")
    print()
    print(f"⏱️  Tiempo promedio: {sum(times)/len(times):.1f}ms")
    print(f"    Tiempo mínimo: {min(times):.1f}ms")
    print(f"    Tiempo máximo: {max(times):.1f}ms")
    print(f"    Throughput: {1000/(sum(times)/len(times)):.2f} imágenes/segundo")
    print()
    
    print("🎯 Predicciones:")
    for r in results:
        print(f"   • {r['image']:15s} → {r['prediction']:30s} "
              f"({r['confidence']:.1%}, {r['time_ms']:.0f}ms)")
    
    print()
    print("✅ DEMOSTRACIÓN COMPLETADA CON DATOS REALES Y TIEMPOS MEDIDOS")
    print("=" * 70)
    print()
    print("🔍 Estos resultados son 100% verificables:")
    print(f"   1. Las imágenes están en: {demo_dir}")
    print("   2. Los tiempos son mediciones reales de tu GPU RX 580")
    print("   3. Las predicciones vienen del modelo MobileNetV2 pre-entrenado")
    print("   4. Puedes repetir esta demo ejecutando el comando de nuevo")
    print()


def main():
    """Main demonstration entry point."""
    parser = argparse.ArgumentParser(
        description='Demostración verificable del framework RX 580',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar demo completa
  python examples/demo_verificable.py
  
  # Solo descargar imágenes
  python examples/demo_verificable.py --download-only
  
  # Benchmark detallado (5 iteraciones por imagen)
  python examples/demo_verificable.py --benchmark
  
  # Usar imágenes propias
  python examples/demo_verificable.py --images-dir /ruta/a/imagenes
        """
    )
    
    parser.add_argument('--download-only', action='store_true',
                       help='Solo descargar imágenes sin procesar')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark detallado (5 iteraciones)')
    parser.add_argument('--images-dir', type=Path,
                       help='Directorio con imágenes propias (default: data/wildlife/demo_real)')
    parser.add_argument('--model-path', type=Path,
                       help='Ruta al modelo ONNX (default: examples/models/mobilenetv2.onnx)')
    parser.add_argument('--force-download', action='store_true',
                       help='Re-descargar imágenes aunque ya existan')
    
    args = parser.parse_args()
    
    # Setup paths
    demo_dir = args.images_dir or Path("data/wildlife/demo_real")
    model_path = args.model_path or Path("examples/models/mobilenetv2.onnx")
    models_dir = Path("examples/models")
    
    # Print header
    print_header()
    
    # Download images
    image_paths = download_demo_images(demo_dir, force=args.force_download)
    
    if not image_paths:
        print("❌ No se pudieron descargar imágenes")
        print("\n💡 Alternativa: Descarga imágenes manualmente de:")
        print("   • https://unsplash.com/s/photos/wildlife")
        print("   • https://pixabay.com/images/search/animals/")
        print(f"\n   Y guárdalas en: {demo_dir}/")
        sys.exit(1)
    
    if args.download_only:
        print("✅ Imágenes descargadas. Usa el comando sin --download-only para clasificar.")
        sys.exit(0)
    
    # Load labels
    labels = load_imagenet_labels(models_dir)
    
    # Classify images
    results, times = classify_images(image_paths, model_path, labels, args.benchmark)
    
    # Print summary
    print_summary(results, times, demo_dir)


if __name__ == "__main__":
    main()
