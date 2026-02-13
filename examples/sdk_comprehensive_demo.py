"""
SDK Comprehensive Demo
======================

This demo showcases all SDK components working together:
1. High-Level API - Quick inference with one-liners
2. Plugin System - Extend functionality
3. Model Registry - Manage local models
4. Model Zoo - Access pre-trained models
5. Builder Pattern - Fluent configuration

Run this demo to see the complete SDK workflow.

Usage:
    python examples/sdk_comprehensive_demo.py

Version: 0.6.0-dev
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.sdk.builder import ConfigBuilder, InferencePipeline, ModelBuilder
from src.sdk.easy import AutoOptimizer, QuickModel, quick_inference
from src.sdk.plugins import Plugin, PluginManager, PluginMetadata, PluginType
from src.sdk.registry import ModelRegistry, ModelTask, ModelZoo


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_high_level_api():
    """Demonstrate high-level API."""
    print_section("1. HIGH-LEVEL API - Quick Inference")

    print("\n✨ One-Liner Inference:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.easy import quick_inference
    
    result = quick_inference("cat.jpg", "mobilenet.onnx")
    print(f"{result.class_name}: {result.confidence:.2%}")
    """)
    print("\nBenefit: Simplest possible API - just provide input and model!")

    print("\n✨ QuickModel Class:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.easy import QuickModel
    
    # Load once, use many times
    model = QuickModel("mobilenet.onnx")
    
    # Single prediction
    result = model.predict("cat.jpg")
    
    # Batch prediction
    results = model.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
    
    # Benchmark performance
    stats = model.benchmark(num_runs=100)
    print(f"Average: {stats['mean_ms']:.2f} ms")
    """)
    print("\nBenefit: Reusable model instance, batch processing, benchmarking!")

    print("\n✨ Auto-Optimization:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.easy import AutoOptimizer
    
    optimizer = AutoOptimizer(target_device="rx580")
    
    # Get suggestions
    suggestions = optimizer.suggest_optimizations("model.onnx")
    for s in suggestions:
        print(f"{s['name']}: {s['expected_speedup']}x speedup")
    
    # Apply optimizations
    optimized = optimizer.optimize("model.onnx")
    """)
    print("\nBenefit: Automatic hardware-specific optimizations!")


def demo_plugin_system():
    """Demonstrate plugin system."""
    print_section("2. PLUGIN SYSTEM - Extend Functionality")

    print("\n🔌 Creating a Plugin:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.plugins import Plugin, PluginMetadata, PluginType
    
    class MyOptimizer(Plugin):
        metadata = PluginMetadata(
            name="my_optimizer",
            version="1.0.0",
            author="Your Name",
            description="Custom optimization technique",
            plugin_type=PluginType.OPTIMIZER
        )
        
        def initialize(self):
            print("Initializing optimizer...")
            return True
        
        def execute(self, model):
            # Your optimization logic here
            optimized_model = optimize(model)
            return optimized_model
        
        def cleanup(self):
            print("Cleaning up...")
            return True
    """)

    print("\n🔌 Using the Plugin Manager:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.plugins import PluginManager
    
    # Initialize manager
    manager = PluginManager()
    manager.discover_plugins()  # Auto-find plugins
    
    # List available plugins
    for name in manager.list_plugins():
        metadata = manager.get_metadata(name)
        print(f"• {name} v{metadata.version}")
        print(f"  {metadata.description}")
    
    # Load and use a plugin
    plugin = manager.load_plugin("my_optimizer")
    result = plugin.execute(model)
    
    # Cleanup
    manager.cleanup_all()
    """)

    # Actually demonstrate
    print("\n📦 Live Demo:")
    print("-" * 80)
    manager = PluginManager()

    discovered = len(manager.list_plugins())
    print(f"✅ Discovered {discovered} plugins")

    if discovered > 0:
        print("\nAvailable plugins:")
        for name in manager.list_plugins():
            metadata = manager.get_metadata(name)
            print(f"  • {name} v{metadata.version}")
            print(f"    Type: {metadata.plugin_type.value}")
            print(f"    {metadata.description}")


def demo_model_registry():
    """Demonstrate model registry."""
    print_section("3. MODEL REGISTRY - Manage Your Models")

    print("\n📝 Registering Models:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.registry import ModelRegistry, ModelTask
    
    registry = ModelRegistry()
    
    # Register a model
    registry.register(
        name="my_classifier",
        path="models/classifier.onnx",
        task=ModelTask.CLASSIFICATION,
        version="2.1.0",
        tags=["int8", "optimized", "production"],
        description="Production classifier v2.1"
    )
    
    # Search models
    results = registry.search(
        task=ModelTask.CLASSIFICATION,
        tags=["int8"]
    )
    
    # Get model info
    metadata = registry.get("my_classifier")
    print(f"Path: {metadata.path}")
    print(f"Size: {metadata.size_mb:.1f} MB")
    
    # Update performance metrics
    registry.update_performance_metrics(
        "my_classifier",
        {"fps": 120, "latency_ms": 8.3, "accuracy": 0.92}
    )
    """)

    print("\nBenefit: Central database of all your models with metadata!")


def demo_model_zoo():
    """Demonstrate model zoo."""
    print_section("4. MODEL ZOO - Pre-trained Models")

    print("\n🦁 Available Models:")
    print("-" * 80)

    zoo = ModelZoo()

    print("\nCLASSIFICATION:")
    class_models = zoo.list_models(task=ModelTask.CLASSIFICATION)
    for name in class_models:
        info = zoo.get_model_info(name)
        print(f"  • {info['display_name']}")
        print(f"    Size: {info['size_mb']:.1f} MB")
        print(f"    Accuracy: {info['accuracy']:.1%}")
        print(f"    RX 580: {info['rx580_fps']} FPS")

    print("\nDETECTION:")
    det_models = zoo.list_models(task=ModelTask.DETECTION)
    for name in det_models:
        info = zoo.get_model_info(name)
        print(f"  • {info['display_name']}")
        print(f"    Size: {info['size_mb']:.1f} MB")
        print(f"    mAP@50: {info['map50']:.3f}")
        print(f"    RX 580: {info['rx580_fps']} FPS")

    print("\n💾 Downloading Models:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.registry import ModelZoo
    
    zoo = ModelZoo()
    
    # Download a model
    model_path = zoo.download("mobilenetv2-int8")
    
    # Use immediately
    from src.sdk.easy import QuickModel
    model = QuickModel(model_path)
    """)

    print("\nBenefit: Pre-optimized models ready to use!")


def demo_builder_pattern():
    """Demonstrate builder pattern."""
    print_section("5. BUILDER PATTERN - Fluent Configuration")

    print("\n🏗️  Inference Pipeline Builder:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.builder import InferencePipeline
    
    pipeline = (InferencePipeline()
        .use_model("mobilenetv2.onnx")
        .on_device("rx580")
        .with_batch_size(32)
        .optimize_for("speed")
        .enable_int8_quantization()
        .add_preprocessing(
            resize=(224, 224),
            normalize=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        .add_postprocessing(
            top_k=5,
            threshold=0.5
        )
        .target_fps(60)
        .max_memory(2000)
        .enable_profiling()
        .build()
    )
    
    # Execute pipeline
    result = pipeline.run("image.jpg")
    """)

    print("\n🏗️  Config Builder:")
    print("-" * 80)
    print("Code:")
    print("""
    from src.sdk.builder import ConfigBuilder
    
    config = (ConfigBuilder()
        .for_task("classification")
        .optimize_for("speed")
        .target_fps(60)
        .max_memory_mb(2000)
        .enable_feature("quantization")
        .enable_feature("profiling")
        .build()
    )
    """)

    # Actually demonstrate
    print("\n📦 Live Demo:")
    print("-" * 80)
    builder = ConfigBuilder()
    config = (
        builder.for_task("classification")
        .optimize_for("speed")
        .target_fps(60)
        .max_memory_mb(2000)
        .enable_feature("quantization")
        .build()
    )

    print("Generated configuration:")
    print(f"  Task: {config['task']}")
    print(f"  Goal: {config['optimization_goal']}")
    print(f"  Target FPS: {config['constraints']['target_fps']}")
    print(f"  Max Memory: {config['constraints']['max_memory_mb']} MB")
    print(f"  Features: {config['features']}")

    print("\nBenefit: Clean, readable, chainable configuration!")


def demo_complete_workflow():
    """Demonstrate complete SDK workflow."""
    print_section("6. COMPLETE WORKFLOW - Everything Together")

    print("\n🚀 Real-World Usage Example:")
    print("-" * 80)
    print("Code:")
    print("""
    # Step 1: Check available models in zoo
    from src.sdk.registry import ModelZoo
    zoo = ModelZoo()
    models = zoo.list_models(task="classification")
    
    # Step 2: Download a model
    model_path = zoo.download("mobilenetv2-int8")
    
    # Step 3: Register it locally
    from src.sdk.registry import ModelRegistry
    registry = ModelRegistry()
    registry.register(
        name="mobile_classifier",
        path=model_path,
        task="classification",
        tags=["production", "int8"]
    )
    
    # Step 4: Create optimized pipeline
    from src.sdk.builder import InferencePipeline
    pipeline = (InferencePipeline()
        .use_model(model_path)
        .on_device("rx580")
        .with_batch_size(32)
        .optimize_for("speed")
        .enable_int8_quantization()
        .add_preprocessing(resize=(224, 224))
        .add_postprocessing(top_k=5)
        .build()
    )
    
    # Step 5: Run inference
    results = pipeline.run("image.jpg")
    
    # Step 6: Update performance metrics
    registry.update_performance_metrics(
        "mobile_classifier",
        {
            "fps": 280,
            "latency_ms": 3.57,
            "accuracy": 0.71
        }
    )
    
    print(f"Prediction: {results['predictions'][0]}")
    print(f"Time: {results['inference_time_ms']:.2f} ms")
    """)

    print("\n✅ Complete Pipeline:")
    print("  1. Discover models → ModelZoo")
    print("  2. Download → zoo.download()")
    print("  3. Register locally → ModelRegistry")
    print("  4. Configure → InferencePipeline (Builder)")
    print("  5. Execute → pipeline.run()")
    print("  6. Track performance → registry.update_performance_metrics()")


def demo_sdk_features_summary():
    """Summary of SDK features."""
    print_section("SDK FEATURES SUMMARY")

    features = {
        "High-Level API": [
            "✅ One-liner inference with quick_inference()",
            "✅ QuickModel class for reusable models",
            "✅ Automatic hardware detection",
            "✅ Built-in benchmarking",
            "✅ Auto-optimization suggestions",
        ],
        "Plugin System": [
            "✅ Extend functionality with custom plugins",
            "✅ Automatic plugin discovery",
            "✅ Plugin lifecycle management",
            "✅ Hook system for events",
            "✅ Multiple plugin types (optimizer, preprocessor, etc.)",
        ],
        "Model Registry": [
            "✅ Centralized model database",
            "✅ Rich metadata (task, version, tags, metrics)",
            "✅ Search and filter capabilities",
            "✅ Performance tracking",
            "✅ Persistent storage (JSON)",
        ],
        "Model Zoo": [
            "✅ Pre-trained models optimized for AMD GPUs",
            "✅ Multiple tasks (classification, detection, etc.)",
            "✅ Performance metrics for RX 580",
            "✅ Easy download and caching",
            "✅ Multiple quantization levels",
        ],
        "Builder Pattern": [
            "✅ Fluent, chainable API",
            "✅ Type-safe configuration",
            "✅ Sensible defaults",
            "✅ IDE auto-completion friendly",
            "✅ Multiple builders (Pipeline, Config, Model)",
        ],
    }

    for category, items in features.items():
        print(f"\n{category}:")
        print("-" * 80)
        for item in items:
            print(f"  {item}")


def main():
    """Run complete SDK demo."""
    print("=" * 80)
    print("  LEGACY GPU AI PLATFORM - SDK COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases all SDK components and their usage.")
    print("Note: Some operations are simulated as they require actual models.")

    # Run all demos
    demo_high_level_api()
    demo_plugin_system()
    demo_model_registry()
    demo_model_zoo()
    demo_builder_pattern()
    demo_complete_workflow()
    demo_sdk_features_summary()

    # Final summary
    print("\n" + "=" * 80)
    print("  DEMO COMPLETE!")
    print("=" * 80)
    print("\n📚 Next Steps:")
    print("  1. Read the API documentation")
    print("  2. Try the examples in your own code")
    print("  3. Create custom plugins for your use case")
    print("  4. Register your models in the registry")
    print("  5. Share your optimizations with the community")
    print("\n🚀 Start building with:")
    print("  from src.sdk.easy import QuickModel")
    print("  model = QuickModel('your_model.onnx')")
    print("  result = model.predict('your_image.jpg')")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
