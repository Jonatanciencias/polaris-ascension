#!/usr/bin/env python3
"""
Model Downloader and Converter

Downloads popular pre-trained models and converts them to ONNX format
for use with the Radeon RX 580 AI Framework.

Supports:
- MobileNetV2 (lightweight classification)
- ResNet-50 (robust classification)
- EfficientNet-B0 (efficient classification)
- YOLOv5s (object detection)

Usage:
    python scripts/download_models.py --all
    python scripts/download_models.py --model resnet50
    python scripts/download_models.py --model yolov5 --size s
"""

import argparse
import sys
from pathlib import Path
import urllib.request
import torch
import torchvision.models as models
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelDownloader:
    """Handles downloading and converting models to ONNX format"""
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize model downloader.
        
        Args:
            models_dir: Directory to save models (default: examples/models)
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "examples" / "models"
        
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Models directory: {self.models_dir}")
    
    def download_mobilenetv2(self) -> Path:
        """
        Download and convert MobileNetV2.
        
        Use case: Lightweight classification for mobile/edge devices
        Parameters: ~3.5M
        Speed: Fast (~500ms on RX 580)
        """
        print("\n" + "="*60)
        print("📦 MobileNetV2 - Lightweight Classification")
        print("="*60)
        
        model_path = self.models_dir / "mobilenetv2.onnx"
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path
        
        print("📥 Downloading MobileNetV2...")
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
        
        print("🔄 Converting to ONNX format...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved: {model_path} ({size_mb:.1f} MB)")
        print(f"   Input: [batch, 3, 224, 224]")
        print(f"   Output: [batch, 1000] (ImageNet classes)")
        print(f"   Best for: Real-time classification, mobile apps")
        
        return model_path
    
    def download_resnet50(self) -> Path:
        """
        Download and convert ResNet-50.
        
        Use case: Robust classification with higher accuracy
        Parameters: ~25M
        Speed: Moderate (~1200ms on RX 580)
        Accuracy: Higher than MobileNetV2
        """
        print("\n" + "="*60)
        print("📦 ResNet-50 - Robust Classification")
        print("="*60)
        
        model_path = self.models_dir / "resnet50.onnx"
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path
        
        print("📥 Downloading ResNet-50...")
        model = models.resnet50(pretrained=True)
        model.eval()
        
        print("🔄 Converting to ONNX format...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved: {model_path} ({size_mb:.1f} MB)")
        print(f"   Input: [batch, 3, 224, 224]")
        print(f"   Output: [batch, 1000] (ImageNet classes)")
        print(f"   Best for: Medical imaging, scientific analysis")
        print(f"   Recommended: Use --fast mode (FP16) for 1.5x speedup")
        
        return model_path
    
    def download_efficientnet_b0(self) -> Path:
        """
        Download and convert EfficientNet-B0.
        
        Use case: Efficient classification with good accuracy/speed trade-off
        Parameters: ~5M
        Speed: Fast (~600ms on RX 580)
        Efficiency: Best accuracy per parameter
        """
        print("\n" + "="*60)
        print("📦 EfficientNet-B0 - Efficient Classification")
        print("="*60)
        
        model_path = self.models_dir / "efficientnet_b0.onnx"
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path
        
        print("📥 Downloading EfficientNet-B0...")
        model = models.efficientnet_b0(pretrained=True)
        model.eval()
        
        print("🔄 Converting to ONNX format...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(model_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved: {model_path} ({size_mb:.1f} MB)")
        print(f"   Input: [batch, 3, 224, 224]")
        print(f"   Output: [batch, 1000] (ImageNet classes)")
        print(f"   Best for: Balance of speed and accuracy")
        print(f"   Recommended: Use --ultra-fast (INT8) for 2.5x speedup")
        
        return model_path
    
    def download_yolov5(self, size: str = 's') -> Path:
        """
        Download YOLOv5 model (pre-converted ONNX).
        
        Use case: Real-time object detection
        Sizes: n (nano), s (small), m (medium), l (large)
        Speed: Fast for detection (~800ms for YOLOv5s on RX 580)
        
        Args:
            size: Model size (n, s, m, l)
        """
        print("\n" + "="*60)
        print(f"📦 YOLOv5{size.upper()} - Object Detection")
        print("="*60)
        
        model_path = self.models_dir / f"yolov5{size}.onnx"
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return model_path
        
        # YOLOv5 ONNX models are available from Ultralytics
        urls = {
            'n': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
            's': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
            'm': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx',
            'l': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.onnx',
        }
        
        if size not in urls:
            print(f"❌ Invalid size: {size}. Choose from: n, s, m, l")
            return None
        
        print(f"📥 Downloading YOLOv5{size.upper()} from Ultralytics...")
        print(f"   URL: {urls[size]}")
        
        try:
            urllib.request.urlretrieve(urls[size], model_path)
            
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ Saved: {model_path} ({size_mb:.1f} MB)")
            print(f"   Input: [batch, 3, 640, 640]")
            print(f"   Output: [batch, 25200, 85] (detections)")
            print(f"   Detects: 80 COCO classes (person, car, dog, etc.)")
            print(f"   Best for: Wildlife monitoring, security, traffic analysis")
            print(f"   Recommended: Use --fast mode for real-time performance")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"   Try downloading manually from: {urls[size]}")
            return None
    
    def download_imagenet_labels(self) -> Path:
        """Download ImageNet class labels."""
        labels_path = self.models_dir / "imagenet_classes.txt"
        
        if labels_path.exists():
            return labels_path
        
        print("\n📥 Downloading ImageNet labels...")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        
        try:
            urllib.request.urlretrieve(url, labels_path)
            print(f"✅ Saved: {labels_path}")
            return labels_path
        except Exception as e:
            print(f"⚠️  Could not download labels: {e}")
            return None
    
    def download_coco_labels(self) -> Path:
        """Download COCO class labels for YOLO."""
        labels_path = self.models_dir / "coco_classes.txt"
        
        if labels_path.exists():
            return labels_path
        
        print("\n📥 Downloading COCO labels...")
        
        # COCO 80 classes
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        with open(labels_path, 'w') as f:
            f.write('\n'.join(coco_classes))
        
        print(f"✅ Saved: {labels_path} (80 classes)")
        return labels_path
    
    def download_all(self):
        """Download all available models."""
        print("\n" + "="*70)
        print("🚀 DOWNLOADING ALL MODELS")
        print("="*70)
        print("\nThis will download:")
        print("  • MobileNetV2 (~14 MB)")
        print("  • ResNet-50 (~98 MB)")
        print("  • EfficientNet-B0 (~20 MB)")
        print("  • YOLOv5s (~14 MB)")
        print("  • Class labels")
        print("\nTotal download: ~150 MB\n")
        
        # Download classification models
        self.download_mobilenetv2()
        self.download_resnet50()
        self.download_efficientnet_b0()
        
        # Download detection model
        self.download_yolov5('s')
        
        # Download labels
        self.download_imagenet_labels()
        self.download_coco_labels()
        
        print("\n" + "="*70)
        print("✅ ALL MODELS DOWNLOADED!")
        print("="*70)
        print(f"\n📁 Models location: {self.models_dir}")
        print("\n💡 Usage:")
        print("   python -m src.cli classify image.jpg --model resnet50 --fast")
        print("   python -m src.cli classify image.jpg --model efficientnet_b0 --ultra-fast")
        print("   python -m src.cli detect image.jpg --model yolov5s --fast")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert AI models for Radeon RX 580',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python scripts/download_models.py --all
  
  # Download specific model
  python scripts/download_models.py --model resnet50
  python scripts/download_models.py --model efficientnet
  python scripts/download_models.py --model yolov5 --size s
  
  # List available models
  python scripts/download_models.py --list
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--model', choices=['mobilenet', 'resnet50', 'efficientnet', 'yolov5'],
                       help='Download specific model')
    parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l'],
                       help='YOLOv5 size (nano, small, medium, large)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--models-dir', type=Path, help='Custom models directory')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available Models:")
        print("\nClassification Models:")
        print("  • mobilenet    - MobileNetV2 (lightweight, ~500ms)")
        print("  • resnet50     - ResNet-50 (robust, ~1200ms)")
        print("  • efficientnet - EfficientNet-B0 (efficient, ~600ms)")
        print("\nObject Detection Models:")
        print("  • yolov5       - YOLOv5 (real-time detection)")
        print("                   Sizes: n (nano), s (small), m (medium), l (large)")
        print("\nAll models support --fast (FP16) and --ultra-fast (INT8) modes")
        print()
        return
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.all:
        downloader.download_all()
    elif args.model:
        if args.model == 'mobilenet':
            downloader.download_mobilenetv2()
        elif args.model == 'resnet50':
            downloader.download_resnet50()
        elif args.model == 'efficientnet':
            downloader.download_efficientnet_b0()
        elif args.model == 'yolov5':
            downloader.download_yolov5(args.size)
        
        # Download labels
        downloader.download_imagenet_labels()
        if args.model == 'yolov5':
            downloader.download_coco_labels()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
