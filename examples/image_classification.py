#!/usr/bin/env python3
"""
Image Classification Example

Demonstrates practical image classification using the RX 580 AI framework.

REAL-WORLD USE CASES:
1. Medical Imaging: Classify X-rays, CT scans in resource-limited clinics
2. Wildlife Conservation: Identify species from camera traps
3. Manufacturing QA: Detect defects in products on assembly lines
4. Agriculture: Identify plant diseases, pest infestations
5. Education: Interactive learning tools for underserved schools
6. Small Business: Automated inventory categorization

This example uses MobileNetV2 - optimized for efficiency on budget hardware.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import urllib.request

import numpy as np
from PIL import Image

from src.core.gpu import GPUManager
from src.core.memory import MemoryManager
from src.inference import InferenceConfig, ONNXInferenceEngine
from src.utils.logging_config import setup_logging

# ImageNet class labels (1000 classes)
IMAGENET_CLASSES = None


def download_imagenet_labels():
    """Download ImageNet class labels"""
    global IMAGENET_CLASSES
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        import json

        with urllib.request.urlopen(url, timeout=10) as response:
            IMAGENET_CLASSES = json.loads(response.read())
        print("✅ Downloaded ImageNet labels")
    except Exception as e:
        print(f"⚠️  Could not download labels: {e}")
        # Use indices as fallback
        IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]


def download_test_model():
    """
    Download a pre-trained MobileNetV2 model in ONNX format.

    MobileNetV2 is chosen because:
    - Lightweight: Only ~14MB
    - Efficient: Optimized for mobile/edge devices
    - Accurate: 71.8% top-1 accuracy on ImageNet
    - Perfect for RX 580: Low memory footprint
    """
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "mobilenetv2.onnx"

    if model_path.exists():
        print(f"✅ Model already downloaded: {model_path}")
        return model_path

    print("📥 Downloading MobileNetV2 model (~14MB)...")
    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"

    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ Model downloaded: {model_path}")
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


def create_test_image():
    """Create a simple test image"""
    img_dir = Path(__file__).parent / "test_images"
    img_dir.mkdir(exist_ok=True)

    # Try to download a sample image
    sample_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/240px-Cat03.jpg"
    )
    img_path = img_dir / "cat.jpg"

    if img_path.exists():
        return img_path

    try:
        print("📥 Downloading test image...")
        urllib.request.urlretrieve(sample_url, img_path)
        print(f"✅ Test image downloaded: {img_path}")
        return img_path
    except Exception as e:
        print(f"⚠️  Could not download test image: {e}")
        # Create a synthetic image
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        img.save(img_path)
        print(f"✅ Created synthetic test image: {img_path}")
        return img_path


def run_classification_demo():
    """
    Run complete classification demo.

    Demonstrates the full pipeline:
    1. Hardware initialization
    2. Model loading
    3. Image preprocessing
    4. Inference
    5. Results interpretation
    """
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("=== RX 580 AI Classification Demo ===")

    # Initialize hardware managers
    logger.info("\n1️⃣  Initializing hardware...")
    gpu_manager = GPUManager()
    gpu_manager.initialize()
    memory_manager = MemoryManager()

    # Print system info
    gpu_info = gpu_manager.get_info()
    if gpu_info:
        logger.info(f"   GPU: {gpu_info.name}")
        logger.info(f"   VRAM: {gpu_info.vram_mb / 1024:.1f}GB")
        logger.info(f"   Backend: {gpu_manager.get_compute_backend()}")

    memory_stats = memory_manager.get_stats()
    logger.info(f"   RAM: {memory_stats.total_ram_gb:.1f}GB")

    # Download resources
    logger.info("\n2️⃣  Preparing resources...")
    download_imagenet_labels()
    model_path = download_test_model()
    test_image = create_test_image()

    # Create inference engine
    logger.info("\n3️⃣  Creating inference engine...")
    config = InferenceConfig(
        device="auto", precision="fp32", batch_size=1, enable_profiling=True, optimization_level=2
    )

    engine = ONNXInferenceEngine(
        config=config, gpu_manager=gpu_manager, memory_manager=memory_manager
    )

    # Load model
    logger.info("\n4️⃣  Loading model...")
    model_info = engine.load_model(model_path)

    # Run inference
    logger.info("\n5️⃣  Running inference...")
    logger.info(f"   Input: {test_image}")

    results = engine.infer(test_image, profile=True)

    # Display results
    logger.info("\n6️⃣  Results:")
    predictions = results["predictions"]
    for i, pred in enumerate(predictions[:5], 1):
        class_id = pred["class_id"]
        confidence = pred["confidence"] * 100
        class_name = IMAGENET_CLASSES[class_id] if IMAGENET_CLASSES else f"class_{class_id}"
        logger.info(f"   {i}. {class_name}: {confidence:.2f}%")

    # Performance statistics
    logger.info("\n7️⃣  Performance:")
    engine.print_performance_stats()

    # Practical applications info
    logger.info("\n" + "=" * 60)
    logger.info("💡 PRACTICAL APPLICATIONS:")
    logger.info("=" * 60)
    logger.info("""
This same technology can be applied to:

🏥 MEDICAL IMAGING:
   - Classify chest X-rays (pneumonia detection)
   - Identify skin lesions (melanoma screening)
   - Analyze CT scans (tumor detection)
   → Enables AI diagnosis in rural clinics with limited budgets

🌍 WILDLIFE CONSERVATION:
   - Identify species from camera trap images
   - Count endangered animals
   - Detect poachers
   → Affordable monitoring for conservation organizations

🏭 MANUFACTURING QA:
   - Detect product defects
   - Classify components on assembly line
   - Quality control automation
   → Small manufacturers can automate without expensive hardware

🌱 AGRICULTURE:
   - Identify plant diseases
   - Detect pest infestations
   - Classify crop health
   → Farmers can get instant diagnosis in the field

📚 EDUCATION:
   - Interactive learning tools
   - Science experiments with real-time classification
   - Accessible AI education
   → Schools in underserved areas can teach AI

🏪 SMALL BUSINESS:
   - Automated inventory categorization
   - Product recognition
   - Document classification
   → Local businesses can leverage AI affordably
    """)

    logger.info("\n✅ Demo complete!")


def run_batch_inference_example():
    """
    Example of batch inference for processing multiple images.
    Useful for processing large datasets or camera feeds.
    """
    logger = setup_logging(level="INFO")
    logger.info("\n=== Batch Inference Example ===")

    # Setup (simplified)
    config = InferenceConfig(device="auto", batch_size=1)
    engine = ONNXInferenceEngine(config=config)

    # Download model
    model_path = download_test_model()
    engine.load_model(model_path)

    # Process multiple images
    test_image = create_test_image()
    images = [test_image] * 5  # Simulate 5 images

    logger.info(f"Processing {len(images)} images...")
    results = engine.batch_infer(images)

    logger.info(f"✅ Processed {len(results)} images")
    logger.info(f"   Average confidence: {np.mean([r['top1_confidence'] for r in results]):.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Image Classification Demo for RX 580 AI Framework"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "batch"],
        default="demo",
        help="Run mode: demo (single image) or batch (multiple images)",
    )

    args = parser.parse_args()

    try:
        if args.mode == "demo":
            run_classification_demo()
        else:
            run_batch_inference_example()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
