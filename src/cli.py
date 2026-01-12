#!/usr/bin/env python3
"""
Radeon RX 580 AI - Command Line Interface

Simple, user-friendly CLI for running AI inference on AMD GPUs.
Designed to be accessible for both technical and non-technical users.

Examples:
    # Basic inference (automatic optimization)
    python -m src.cli classify image.jpg
    
    # Fast mode (FP16, ~1.5x speedup)
    python -m src.cli classify image.jpg --fast
    
    # Maximum speed (INT8, ~2.5x speedup)
    python -m src.cli classify image.jpg --ultra-fast
    
    # Batch processing multiple images
    python -m src.cli classify folder/*.jpg --batch 4
    
    # Get system information
    python -m src.cli info
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

from .core.gpu import GPUManager
from .core.memory import MemoryManager
from .inference.onnx_engine import ONNXInferenceEngine
from .inference.base import InferenceConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class CLI:
    """Command Line Interface for Radeon RX 580 AI Framework"""
    
    def __init__(self):
        self.gpu_manager = GPUManager()
        self.memory_manager = MemoryManager()
    
    def info(self):
        """Display system information"""
        print("\n" + "="*60)
        print("🖥️  RADEON RX 580 AI - SYSTEM INFORMATION")
        print("="*60)
        
        # GPU Info
        gpu_info = self.gpu_manager.get_info()
        print(f"\n📊 GPU Information:")
        if gpu_info:
            print(f"   Name: {gpu_info.name}")
            print(f"   Driver: {gpu_info.driver}")
            print(f"   VRAM: {gpu_info.vram_gb:.1f} GB")
            print(f"   OpenCL: {'✅ Available' if gpu_info.opencl_available else '❌ Not available'}")
        else:
            print(f"   ⚠️  GPU not detected (will use CPU)")
        
        # Memory Info
        memory_stats = self.memory_manager.get_stats()
        
        print(f"\n💾 Memory Information:")
        vram_gb = memory_stats.gpu_vram_gb if memory_stats.gpu_vram_gb else 0
        print(f"   VRAM: {vram_gb:.1f} GB total")
        print(f"   RAM: {memory_stats.available_ram_gb:.1f} GB available / {memory_stats.total_ram_gb:.1f} GB total")
        
        # Optimization Recommendations
        print(f"\n⚡ Performance Tips:")
        print(f"   • Use --fast for 1.5x speedup (FP16)")
        print(f"   • Use --ultra-fast for 2.5x speedup (INT8)")
        print(f"   • Use --batch N for processing multiple images")
        print(f"   • Available VRAM allows batch size up to ~8")
        
        print("\n" + "="*60 + "\n")
    
    def classify(self, 
                 image_paths: List[str],
                 model_path: Optional[str] = None,
                 fast: bool = False,
                 ultra_fast: bool = False,
                 batch_size: int = 1,
                 top_k: int = 5,
                 output: Optional[str] = None):
        """
        Classify one or more images.
        
        Args:
            image_paths: List of image file paths or patterns
            model_path: Path to ONNX model (defaults to MobileNetV2)
            fast: Use FP16 precision (~1.5x speedup)
            ultra_fast: Use INT8 precision (~2.5x speedup)
            batch_size: Number of images to process together
            top_k: Number of top predictions to show
            output: Optional output file for results (JSON/CSV)
        """
        # Determine precision
        if ultra_fast:
            precision = 'int8'
            mode_name = "Ultra-Fast Mode (INT8)"
        elif fast:
            precision = 'fp16'
            mode_name = "Fast Mode (FP16)"
        else:
            precision = 'fp32'
            mode_name = "Standard Mode (FP32)"
        
        # Setup configuration
        config = InferenceConfig(
            device='auto',
            precision=precision,
            batch_size=batch_size,
            enable_profiling=True,
            optimization_level=2
        )
        
        # Create engine
        print(f"\n🚀 Initializing {mode_name}...")
        engine = ONNXInferenceEngine(config, self.gpu_manager, self.memory_manager)
        
        # Load model
        if model_path is None:
            # Default to MobileNetV2
            model_path = Path(__file__).parent.parent / "examples/models/mobilenetv2.onnx"
            if not model_path.exists():
                print(f"❌ Error: Default model not found at {model_path}")
                print(f"   Please specify a model with --model or download MobileNetV2")
                return
        
        print(f"📦 Loading model from {model_path}...")
        engine.load_model(model_path)
        
        # Show optimization info
        opt_info = engine.get_optimization_info()
        print(f"\n⚙️  Optimization Settings:")
        print(f"   Precision: {opt_info['precision'].upper()}")
        print(f"   Batch Size: {opt_info['batch_size']}")
        print(f"   Expected Performance: {opt_info['expected_speedup']}")
        print(f"   Memory Savings: {opt_info['memory_savings']}")
        print(f"   Accuracy: {opt_info['accuracy']}")
        
        # Process images
        print(f"\n🖼️  Processing {len(image_paths)} image(s)...\n")
        
        if len(image_paths) == 1:
            # Single image
            result = engine.infer(image_paths[0])
            self._print_result(image_paths[0], result, top_k)
        else:
            # Batch processing
            results = engine.infer_batch(image_paths, batch_size)
            for img_path, result in zip(image_paths, results):
                self._print_result(img_path, result, top_k)
        
        # Show performance stats
        if engine.profiler:
            stats = engine.profiler.get_statistics()
            print(f"\n📈 Performance Statistics:")
            print(f"   Average Inference Time: {stats['mean']:.1f}ms")
            print(f"   Throughput: {1000/stats['mean']:.1f} images/second")
            if len(image_paths) > 1:
                print(f"   Total Time: {stats['total']:.1f}ms for {len(image_paths)} images")
        
        print()
    
    def _print_result(self, image_path: str, result: dict, top_k: int = 5):
        """Print classification results in a user-friendly format"""
        print(f"📸 {Path(image_path).name}")
        print(f"   Top prediction: Class {result['top1_class']} ({result['top1_confidence']:.1%} confident)")
        
        if top_k > 1 and 'predictions' in result:
            print(f"   Top {top_k} predictions:")
            for i, pred in enumerate(result['predictions'][:top_k], 1):
                print(f"      {i}. Class {pred['class_id']}: {pred['confidence']:.1%}")
        print()
    
    def benchmark(self, 
                  model_path: Optional[str] = None,
                  iterations: int = 100):
        """
        Run performance benchmark comparing different optimization modes.
        
        Args:
            model_path: Path to ONNX model
            iterations: Number of iterations per mode
        """
        print("\n" + "="*60)
        print("🔬 PERFORMANCE BENCHMARK")
        print("="*60)
        
        if model_path is None:
            model_path = Path(__file__).parent.parent / "examples/models/mobilenetv2.onnx"
        
        # Test image
        test_image = Path(__file__).parent.parent / "examples/test_images"
        if test_image.exists():
            test_images = list(test_image.glob("*.jpg")) + list(test_image.glob("*.png"))
            if test_images:
                test_image = test_images[0]
        
        modes = [
            ('fp32', 'Standard (FP32)'),
            ('fp16', 'Fast (FP16)'),
            ('int8', 'Ultra-Fast (INT8)')
        ]
        
        results = {}
        
        for precision, name in modes:
            print(f"\n⚡ Testing {name}...")
            
            config = InferenceConfig(
                precision=precision,
                enable_profiling=True,
                optimization_level=2
            )
            
            engine = ONNXInferenceEngine(config, self.gpu_manager, self.memory_manager)
            engine.load_model(model_path)
            
            # Warmup
            for _ in range(10):
                engine.infer(test_image)
            
            # Benchmark
            engine.profiler.reset()
            for _ in range(iterations):
                engine.infer(test_image)
            
            stats = engine.profiler.get_statistics()
            results[precision] = stats
            
            print(f"   Average: {stats['mean']:.1f}ms")
            print(f"   FPS: {1000/stats['mean']:.1f}")
        
        # Summary
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"   {'Mode':<20} {'Time':<15} {'FPS':<10} {'Speedup'}")
        print(f"   {'-'*20} {'-'*15} {'-'*10} {'-'*10}")
        
        baseline = results['fp32']['mean']
        for precision, name in modes:
            stats = results[precision]
            speedup = baseline / stats['mean']
            fps = 1000 / stats['mean']
            print(f"   {name:<20} {stats['mean']:.1f}ms{'':<9} {fps:.1f}{'':<5} {speedup:.2f}x")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Radeon RX 580 AI - Accessible AI inference on AMD GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get system information
  python -m src.cli info
  
  # Classify a single image (standard quality)
  python -m src.cli classify photo.jpg
  
  # Fast mode (~1.5x speedup, great for real-time)
  python -m src.cli classify photo.jpg --fast
  
  # Ultra-fast mode (~2.5x speedup, for high throughput)
  python -m src.cli classify photo.jpg --ultra-fast
  
  # Process multiple images in batch
  python -m src.cli classify image1.jpg image2.jpg image3.jpg --batch 4
  
  # Run performance benchmark
  python -m src.cli benchmark
  
For more information, visit: https://github.com/yourusername/radeon-rx580-ai
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    subparsers.add_parser('info', help='Display system information')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify images')
    classify_parser.add_argument('images', nargs='+', help='Image file path(s)')
    classify_parser.add_argument('--model', '-m', help='Path to ONNX model')
    classify_parser.add_argument('--fast', '-f', action='store_true', 
                                 help='Fast mode (FP16, ~1.5x speedup)')
    classify_parser.add_argument('--ultra-fast', '-u', action='store_true',
                                 help='Ultra-fast mode (INT8, ~2.5x speedup)')
    classify_parser.add_argument('--batch', '-b', type=int, default=1,
                                 help='Batch size for processing multiple images')
    classify_parser.add_argument('--top-k', '-k', type=int, default=5,
                                 help='Number of top predictions to show')
    classify_parser.add_argument('--output', '-o', help='Output file (JSON/CSV)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--model', '-m', help='Path to ONNX model')
    benchmark_parser.add_argument('--iterations', '-i', type=int, default=100,
                                  help='Number of iterations per mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    cli = CLI()
    
    if args.command == 'info':
        cli.info()
    elif args.command == 'classify':
        cli.classify(
            image_paths=args.images,
            model_path=args.model,
            fast=args.fast,
            ultra_fast=args.ultra_fast,
            batch_size=args.batch,
            top_k=args.top_k,
            output=args.output
        )
    elif args.command == 'benchmark':
        cli.benchmark(model_path=args.model, iterations=args.iterations)


if __name__ == '__main__':
    main()
