"""
Create simple ONNX models for testing

Creates tiny test models to validate the loader functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

print("Creating test ONNX models...")

try:
    import torch
    import torch.nn as nn
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Model 1: Simple classifier (like MobileNet structure)
    print("\n1. Creating simple classifier...")
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = SimpleClassifier()
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = models_dir / "simple_classifier.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"   ✅ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    # Model 2: Tiny detector (simplified YOLO-like)
    print("\n2. Creating tiny detector...")
    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Conv2d(16, 5, 1)  # 5 outputs: x, y, w, h, conf
        
        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x
    
    model = TinyDetector()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 416, 416)
    output_path = models_dir / "tiny_detector.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['detections'],
        dynamic_axes={'image': {0: 'batch_size'}, 'detections': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"   ✅ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    # Model 3: Micro model (for testing)
    print("\n3. Creating micro model...")
    class MicroModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = MicroModel()
    model.eval()
    
    dummy_input = torch.randn(1, 10)
    output_path = models_dir / "micro_model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"   ✅ Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    print("\n✅ All test models created successfully!")
    print(f"\nModels directory: {models_dir}")
    print("You can now test with these models:")
    print("  - simple_classifier.onnx: 3x224x224 -> 10 classes")
    print("  - tiny_detector.onnx: 3x416x416 -> detections")
    print("  - micro_model.onnx: 10 -> 5 (for quick tests)")
    
except ImportError:
    print("⚠️  PyTorch not installed. Cannot create ONNX models.")
    print("Install with: pip install torch")
except Exception as e:
    print(f"❌ Error creating models: {e}")
