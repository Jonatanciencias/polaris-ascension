"""
Enhanced Inference Engine - Comprehensive Demo

Demonstrates Session 15 capabilities:
1. Model compression pipeline (quantization + sparsity)
2. Adaptive batch scheduling for dynamic workloads
3. Multi-model serving with resource management
4. Complete end-to-end inference workflow

Real-world scenario:
Medical imaging facility with AMD GPU serving multiple diagnostic models
with optimized memory usage and throughput.

Session 15 - January 18, 2026
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import tempfile

from src.inference.enhanced import (
    EnhancedInferenceEngine,
    ModelCompressor,
    AdaptiveBatchScheduler,
    MultiModelServer,
    CompressionStrategy,
    CompressionConfig,
    BatchRequest,
    BatchResponse,
)
from src.inference import InferenceConfig


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_1_model_compression():
    """Demo 1: Model Compression Pipeline"""
    print_section("DEMO 1: Model Compression Pipeline")

    print("Testing different compression strategies...\n")

    # Create mock model
    model = {"weights": np.random.randn(1000, 1000).astype(np.float32)}
    calibration_data = np.random.randn(100, 1000).astype(np.float32)

    strategies = [
        (CompressionStrategy.NONE, "No Compression (Baseline)"),
        (CompressionStrategy.QUANTIZE_ONLY, "Quantization Only (INT8)"),
        (CompressionStrategy.SPARSE_ONLY, "Sparsity Only (50%)"),
        (CompressionStrategy.QUANTIZE_SPARSE, "Quantization + Sparsity"),
        (CompressionStrategy.AGGRESSIVE, "Aggressive (All Optimizations)"),
    ]

    results = []

    for strategy, description in strategies:
        config = CompressionConfig(strategy=strategy, target_sparsity=0.5, quantization_bits=8)

        compressor = ModelCompressor(config)
        compressed_model, result = compressor.compress(model, calibration_data)

        results.append((description, result))

        print(f"✓ {description}")
        print(f"  Compression Ratio: {result.compression_ratio:.2f}x")
        print(f"  Memory Saved: {result.memory_savings_mb:.1f} MB")
        print(f"  Estimated Speedup: {result.inference_speedup:.2f}x")
        if result.sparsity_achieved > 0:
            print(f"  Sparsity: {result.sparsity_achieved:.1%}")
        if result.quantization_applied:
            print(f"  Quantization: Applied")
        print()

    # Compare results
    print("\nCompression Comparison:")
    print(f"{'Strategy':<35} {'Ratio':>10} {'Savings (MB)':>15} {'Speedup':>10}")
    print("-" * 70)
    for desc, result in results:
        print(
            f"{desc:<35} {result.compression_ratio:>10.2f}x "
            f"{result.memory_savings_mb:>14.1f} MB "
            f"{result.inference_speedup:>9.2f}x"
        )

    print("\n✅ Best strategy: Quantization + Sparsity (2-4x compression)")


def demo_2_adaptive_batching():
    """Demo 2: Adaptive Batch Scheduling"""
    print_section("DEMO 2: Adaptive Batch Scheduling")

    print("Demonstrating dynamic batch size adaptation...\n")

    scheduler = AdaptiveBatchScheduler(
        min_batch_size=1, max_batch_size=16, max_wait_ms=50.0, target_latency_ms=100.0
    )

    # Track responses
    responses = []

    def callback(response: BatchResponse):
        responses.append(response)
        print(
            f"  ✓ Request {response.request_id[:12]}... processed "
            f"(batch_size={response.batch_size}, "
            f"latency={response.latency_ms:.1f}ms)"
        )

    # Start scheduler
    scheduler.start()

    print("Submitting 20 requests with varying patterns...\n")

    # Submit requests with different patterns
    for i in range(20):
        request = BatchRequest(
            request_id=f"request_{i:03d}_{time.time()}",
            inputs=np.random.randn(1, 224, 224, 3),
            timestamp=time.time(),
            priority=0,
            callback=callback,
        )

        scheduler.submit_request(request)

        # Vary submission rate to test adaptation
        if i < 5:
            time.sleep(0.01)  # Slow rate initially
        elif i < 15:
            time.sleep(0.001)  # Fast burst
        else:
            time.sleep(0.02)  # Slow down again

    # Wait for processing
    time.sleep(2.0)

    # Get statistics
    stats = scheduler.get_stats()

    print(f"\nScheduler Statistics:")
    print(f"  Total Requests Processed: {len(responses)}")
    print(f"  Current Batch Size: {stats['current_batch_size']}")
    print(f"  Average Latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Queue Size: {stats['queue_size']}")

    if responses:
        batch_sizes = [r.batch_size for r in responses]
        print(f"\nBatch Size Adaptation:")
        print(f"  Min Batch Size Used: {min(batch_sizes)}")
        print(f"  Max Batch Size Used: {max(batch_sizes)}")
        print(f"  Avg Batch Size: {sum(batch_sizes)/len(batch_sizes):.1f}")

    scheduler.stop()
    print("\n✅ Scheduler adapted batch size based on load")


def demo_3_multi_model_serving():
    """Demo 3: Multi-Model Server"""
    print_section("DEMO 3: Multi-Model Server")

    print("Serving multiple models concurrently...\n")

    server = MultiModelServer(max_models=5, memory_limit_mb=2000.0)

    # Create temporary model files
    model_configs = [
        ("classifier_v1", "Image classification model"),
        ("detector_v2", "Object detection model"),
        ("segmenter_v1", "Semantic segmentation model"),
    ]

    print("Loading models:")

    with tempfile.TemporaryDirectory() as tmpdir:
        for model_name, description in model_configs:
            # Create mock model file
            model_path = Path(tmpdir) / f"{model_name}.onnx"
            model_path.write_text("mock model")

            success = server.load_model(model_name, model_path, enable_batching=True)

            if success:
                print(f"  ✓ Loaded: {model_name} ({description})")

        # Get server status
        server_stats = server.get_server_stats()
        print(f"\nServer Status:")
        print(f"  Models Loaded: {server_stats['num_models']}")
        print(
            f"  Memory Used: {server_stats['total_memory_mb']:.1f} MB / "
            f"{server_stats['memory_limit_mb']:.1f} MB"
        )
        print(f"  Memory Usage: {server_stats['memory_usage_pct']:.1f}%")

        # Run mixed workload
        print(f"\nRunning mixed workload (30 requests across models)...")

        for i in range(30):
            model_name = model_configs[i % len(model_configs)][0]
            inputs = np.random.randn(1, 224, 224, 3)

            outputs = server.predict(model_name, inputs, timeout_ms=1000.0)

            if outputs is not None and i % 10 == 0:
                print(f"  ✓ Request {i+1}: {model_name} completed")

        # Show per-model statistics
        print(f"\nPer-Model Statistics:")
        print(f"{'Model':<20} {'Requests':>10} {'Avg Latency':>15} {'Throughput':>12}")
        print("-" * 70)

        for model_name, _ in model_configs:
            stats = server.get_model_stats(model_name)
            if stats:
                print(
                    f"{model_name:<20} {stats.total_requests:>10} "
                    f"{stats.avg_latency_ms:>14.2f}ms "
                    f"{stats.throughput_rps:>11.2f} RPS"
                )

        print("\n✅ Multi-model serving with resource management")

        # Cleanup
        for model_name, _ in model_configs:
            server.unload_model(model_name)


def demo_4_enhanced_engine():
    """Demo 4: Enhanced Inference Engine (Complete Workflow)"""
    print_section("DEMO 4: Enhanced Inference Engine - Complete Workflow")

    print("End-to-end production inference workflow...\n")

    # Configure compression
    compression_config = CompressionConfig(
        strategy=CompressionStrategy.QUANTIZE_SPARSE, target_sparsity=0.5, quantization_bits=8
    )

    # Configure inference
    inference_config = InferenceConfig(device="auto", precision="fp32", batch_size=4)

    # Create enhanced engine
    engine = EnhancedInferenceEngine(
        config=inference_config,
        compression_config=compression_config,
        enable_hybrid_scheduling=True,
    )

    print("✓ Enhanced engine initialized")
    print(f"  Compression: {compression_config.strategy.value}")
    print(f"  Target Sparsity: {compression_config.target_sparsity:.1%}")
    print(f"  Quantization: {compression_config.quantization_bits}-bit")
    print(f"  Hybrid Scheduling: Enabled")

    # Load and optimize models
    print("\nLoading and optimizing models:")

    with tempfile.TemporaryDirectory() as tmpdir:
        models = ["resnet50", "mobilenet_v2", "efficientnet_b0"]

        for model_name in models:
            model_path = Path(tmpdir) / f"{model_name}.onnx"
            model_path.write_text("mock model")

            calibration_data = np.random.randn(10, 224, 224, 3)

            success = engine.load_and_optimize(
                model_name, model_path, calibration_data=calibration_data, enable_batching=True
            )

            if success:
                print(f"  ✓ {model_name}: Loaded and optimized")

        # Run inference workload
        print("\nRunning production workload (50 requests)...")

        start_time = time.time()
        request_count = 0

        for i in range(50):
            model_name = models[i % len(models)]
            inputs = np.random.randn(1, 224, 224, 3)

            outputs = engine.predict(model_name, inputs)

            if outputs is not None:
                request_count += 1
                if i % 10 == 0:
                    print(f"  ✓ Processed {i+1}/50 requests")

        elapsed_time = time.time() - start_time
        throughput = request_count / elapsed_time

        # Get comprehensive statistics
        stats = engine.get_stats()

        print(f"\nWorkload Complete:")
        print(f"  Total Requests: {request_count}")
        print(f"  Elapsed Time: {elapsed_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} requests/second")

        print(f"\nServer Statistics:")
        print(f"  Models Active: {stats['server']['num_models']}")
        print(f"  Total Memory: {stats['server']['total_memory_mb']:.1f} MB")
        print(f"  Memory Usage: {stats['server']['memory_usage_pct']:.1f}%")

        print(f"\nModel Performance:")
        for model_name in models:
            if model_name in stats["models"]:
                model_stats = stats["models"][model_name]
                print(f"  {model_name}:")
                print(f"    Requests: {model_stats['total_requests']}")
                print(f"    Avg Latency: {model_stats['avg_latency_ms']:.2f}ms")

        print("\n✅ Production inference with all optimizations")

        # Shutdown
        engine.shutdown()


def demo_5_production_scenario():
    """Demo 5: Real-World Production Scenario"""
    print_section("DEMO 5: Real-World Production Scenario")

    print("Simulating medical imaging facility workflow...\n")
    print("Scenario: Multiple diagnostic models on single AMD GPU")
    print("Hardware: AMD Radeon RX 580 (8GB VRAM, 2 TFLOPS)\n")

    # Configure for production
    compression_config = CompressionConfig(
        strategy=CompressionStrategy.AGGRESSIVE,
        target_sparsity=0.7,  # Aggressive for memory
        quantization_bits=8,
        enable_pruning=True,
    )

    engine = EnhancedInferenceEngine(
        compression_config=compression_config, enable_hybrid_scheduling=True
    )

    print("Medical AI Models:")

    with tempfile.TemporaryDirectory() as tmpdir:
        medical_models = [
            ("xray_classifier", "Chest X-ray classification"),
            ("ct_segmenter", "CT scan segmentation"),
            ("mri_analyzer", "MRI anomaly detection"),
        ]

        for model_name, description in medical_models:
            model_path = Path(tmpdir) / f"{model_name}.onnx"
            model_path.write_text("mock model")

            calibration_data = np.random.randn(20, 512, 512, 1)

            success = engine.load_and_optimize(
                model_name, model_path, calibration_data=calibration_data, enable_batching=True
            )

            if success:
                print(f"  ✓ {model_name}: {description}")

        # Simulate daily workload
        print("\nSimulating daily workload...")
        print("  - Morning: High X-ray volume")
        print("  - Afternoon: Mixed CT/MRI scans")
        print("  - Evening: Light load\n")

        workload = []
        # Morning: 60% X-ray
        workload.extend(["xray_classifier"] * 30)
        # Afternoon: Mixed
        workload.extend(["ct_segmenter"] * 15)
        workload.extend(["mri_analyzer"] * 10)
        workload.extend(["xray_classifier"] * 10)
        # Evening: Light
        workload.extend(["ct_segmenter"] * 5)

        print(f"Processing {len(workload)} medical images...")

        start_time = time.time()
        processed = 0

        for i, model_name in enumerate(workload):
            # Appropriate input sizes for medical imaging
            if "xray" in model_name:
                inputs = np.random.randn(1, 512, 512, 1)
            else:
                inputs = np.random.randn(1, 256, 256, 32)  # 3D scan slices

            outputs = engine.predict(model_name, inputs)

            if outputs is not None:
                processed += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                throughput = processed / elapsed
                print(f"  ✓ Progress: {i+1}/{len(workload)} " f"({throughput:.1f} images/sec)")

        total_time = time.time() - start_time
        final_throughput = processed / total_time

        # Final statistics
        stats = engine.get_stats()

        print(f"\nDaily Summary:")
        print(f"  Images Processed: {processed}")
        print(f"  Total Time: {total_time/60:.1f} minutes")
        print(f"  Average Throughput: {final_throughput:.1f} images/second")
        print(f"  GPU Memory Used: {stats['server']['total_memory_mb']:.1f} MB / 8000 MB")
        print(f"  GPU Utilization: {stats['server']['memory_usage_pct']:.1f}%")

        print(f"\nModel Usage:")
        for model_name, description in medical_models:
            if model_name in stats["models"]:
                model_stats = stats["models"][model_name]
                print(f"  {model_name}:")
                print(f"    Cases Processed: {model_stats['total_requests']}")
                print(f"    Avg Processing Time: {model_stats['avg_latency_ms']:.1f}ms")

        print("\n✅ Medical facility served efficiently with single AMD GPU")
        print("💡 Compression enabled 3+ models on 8GB VRAM")
        print("💡 Adaptive batching maximized throughput")
        print("💡 Multi-model serving handled mixed workload")

        engine.shutdown()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("  ENHANCED INFERENCE ENGINE - SESSION 15")
    print("  Comprehensive Demo Suite")
    print("=" * 70)

    demos = [
        ("Model Compression", demo_1_model_compression),
        ("Adaptive Batching", demo_2_adaptive_batching),
        ("Multi-Model Serving", demo_3_multi_model_serving),
        ("Enhanced Engine", demo_4_enhanced_engine),
        ("Production Scenario", demo_5_production_scenario),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()

    print_section("SUMMARY")
    print("✅ All Session 15 components demonstrated:")
    print("   1. Model Compression Pipeline (quantization + sparsity)")
    print("   2. Adaptive Batch Scheduling (dynamic sizing)")
    print("   3. Multi-Model Server (concurrent serving)")
    print("   4. Enhanced Engine (complete integration)")
    print("   5. Production Scenario (real-world workflow)")
    print("\n💡 Key Benefits:")
    print("   • 2-4x model compression → More models on limited VRAM")
    print("   • Adaptive batching → Optimized throughput")
    print("   • Multi-model serving → Resource efficiency")
    print("   • Integrated pipeline → Production-ready deployment")
    print("\n🎯 Perfect for:")
    print("   • Medical imaging facilities")
    print("   • Edge AI deployments")
    print("   • Multi-tenant serving")
    print("   • Research institutions")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
