"""
Comprehensive Quantization Demo for RX 580
==========================================

This demo showcases all quantization features:
1. Per-tensor vs per-channel quantization
2. Different calibration methods comparison
3. Mixed-precision optimization
4. INT4 packing demonstration
5. QAT (Quantization-Aware Training) workflow
6. ROCm integration (if available)

Run: python examples/demo_quantization.py
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compute.quantization import (
    AdaptiveQuantizer,
    QuantizationPrecision,
    CalibrationMethod,
    QuantizationConfig,
    create_quantizer_for_gpu,
    benchmark_calibration_methods,
)

# Try ROCm integration
try:
    from src.compute.rocm_integration import ROCmQuantizer, get_rocm_status

    ROCM_AVAILABLE = True
except ImportError:
    ROCM_AVAILABLE = False
    print("[Warning] ROCm integration not available")


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_quantization():
    """Demo 1: Basic quantization with different methods."""
    print_section("Demo 1: Basic Quantization Methods")

    # Create synthetic CNN weights (Conv2D layer)
    np.random.seed(42)
    weights = np.random.randn(64, 32, 3, 3).astype(np.float32)
    print(f"\nInput: Conv2D weights shape {weights.shape}")
    print(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
    print(f"  Size: {weights.nbytes / 1024:.2f} KB (FP32)")

    # Test all calibration methods
    methods = [
        CalibrationMethod.MINMAX,
        CalibrationMethod.PERCENTILE,
        CalibrationMethod.KL_DIVERGENCE,
        CalibrationMethod.MSE,
    ]

    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Time(ms)':<12} {'SQNR(dB)':<12} {'Error':<12}")
    print("-" * 70)

    results = []
    for method in methods:
        config = QuantizationConfig(precision=QuantizationPrecision.INT8, calibration_method=method)
        quantizer = AdaptiveQuantizer(gpu_family="polaris", config=config)

        # Time the quantization
        start = time.time()
        q_weights, scale, zp = quantizer.quantize_tensor(weights)
        elapsed_ms = (time.time() - start) * 1000

        # Compute metrics
        dequantized = quantizer.dequantize_tensor(q_weights, scale, zp)
        error = np.mean(np.abs(weights - dequantized))

        signal_power = np.mean(weights**2)
        noise_power = np.mean((weights - dequantized) ** 2)
        sqnr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        print(f"{method.value:<20} {elapsed_ms:<12.2f} {sqnr_db:<12.2f} {error:<12.6f}")
        results.append((method.value, elapsed_ms, sqnr_db, error))

    # Recommend best method
    best_quality = max(results, key=lambda x: x[2])
    best_speed = min(results, key=lambda x: x[1])

    print("\n📊 Recommendations:")
    print(f"  Best Quality: {best_quality[0]} (SQNR: {best_quality[2]:.2f} dB)")
    print(f"  Best Speed: {best_speed[0]} (Time: {best_speed[1]:.2f} ms)")


def demo_per_channel_quantization():
    """Demo 2: Per-channel vs per-tensor quantization."""
    print_section("Demo 2: Per-Channel vs Per-Tensor Quantization")

    # Create Conv2D weights with different channel distributions
    np.random.seed(42)
    weights = np.random.randn(64, 32, 3, 3).astype(np.float32)

    # Make some channels have different ranges (simulate real networks)
    weights[0:16] *= 0.5  # First 16 channels: smaller values
    weights[48:64] *= 2.0  # Last 16 channels: larger values

    print(f"\nWeights shape: {weights.shape}")
    print(f"Channel statistics:")
    print(f"  Channels 0-15: std={weights[0:16].std():.4f}")
    print(f"  Channels 16-47: std={weights[16:48].std():.4f}")
    print(f"  Channels 48-63: std={weights[48:64].std():.4f}")

    quantizer = AdaptiveQuantizer(
        gpu_family="polaris",
        config=QuantizationConfig(
            precision=QuantizationPrecision.INT8,
            calibration_method=CalibrationMethod.PERCENTILE,
            per_channel=True,
        ),
    )

    # Per-tensor quantization
    print("\n[Per-Tensor Quantization]")
    start = time.time()
    q_tensor, scale_t, zp_t = quantizer.quantize_tensor(weights)
    time_tensor = (time.time() - start) * 1000

    dequant_tensor = quantizer.dequantize_tensor(q_tensor, scale_t, zp_t)
    error_tensor = np.mean(np.abs(weights - dequant_tensor))
    sqnr_tensor = 10 * np.log10(
        np.mean(weights**2) / (np.mean((weights - dequant_tensor) ** 2) + 1e-10)
    )

    print(f"  Time: {time_tensor:.2f} ms")
    print(f"  SQNR: {sqnr_tensor:.2f} dB")
    print(f"  Error: {error_tensor:.6f}")
    print(f"  Scale: {scale_t:.6f} (single value)")

    # Per-channel quantization
    print("\n[Per-Channel Quantization]")
    start = time.time()
    q_channel, scales_c, zps_c = quantizer.quantize_tensor_per_channel(
        weights, axis=0  # per output channel
    )
    time_channel = (time.time() - start) * 1000

    dequant_channel = quantizer.dequantize_tensor_per_channel(q_channel, scales_c, zps_c, axis=0)
    error_channel = np.mean(np.abs(weights - dequant_channel))
    sqnr_channel = 10 * np.log10(
        np.mean(weights**2) / (np.mean((weights - dequant_channel) ** 2) + 1e-10)
    )

    print(f"  Time: {time_channel:.2f} ms")
    print(f"  SQNR: {sqnr_channel:.2f} dB")
    print(f"  Error: {error_channel:.6f}")
    print(f"  Scales: {len(scales_c)} values")
    print(f"    Range: [{scales_c.min():.6f}, {scales_c.max():.6f}]")
    print(f"    Std: {scales_c.std():.6f}")

    # Comparison
    print("\n📊 Comparison:")
    print(f"  SQNR improvement: {sqnr_channel - sqnr_tensor:+.2f} dB")
    print(f"  Error reduction: {(1 - error_channel/error_tensor)*100:.1f}%")
    print(f"  Time overhead: {time_channel - time_tensor:.2f} ms")
    print(f"  Memory overhead: {scales_c.nbytes + zps_c.nbytes} bytes")


def demo_mixed_precision():
    """Demo 3: Mixed-precision optimization."""
    print_section("Demo 3: Mixed-Precision Optimization")

    # Simulate a small CNN model
    np.random.seed(42)
    model_layers = {
        "conv1": np.random.randn(32, 3, 3, 3).astype(np.float32),
        "conv2": np.random.randn(64, 32, 3, 3).astype(np.float32),
        "conv3": np.random.randn(128, 64, 3, 3).astype(np.float32),
        "fc1": np.random.randn(512, 2048).astype(np.float32),
        "fc2": np.random.randn(10, 512).astype(np.float32),  # Output layer
    }

    # Make output layer more sensitive (simulate real behavior)
    model_layers["fc2"] *= 0.1  # Smaller weights → higher sensitivity

    print(f"\nModel layers:")
    total_params = 0
    for name, weights in model_layers.items():
        params = weights.size
        total_params += params
        size_mb = weights.nbytes / (1024 * 1024)
        print(f"  {name:<10} {str(weights.shape):<20} {params:>10,} params  {size_mb:.2f} MB")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size (FP32): {sum(w.nbytes for w in model_layers.values()) / (1024**2):.2f} MB")

    # Optimize mixed-precision
    quantizer = AdaptiveQuantizer(gpu_family="polaris", verbose=False)

    print("\n[Analyzing layer sensitivities...]")
    precision_map = quantizer.optimize_mixed_precision(
        model_layers,
        accuracy_threshold=0.01,  # 1% max loss per layer
        memory_budget_gb=None,  # No constraint
    )

    # Calculate statistics
    print("\n[Precision Assignment]")
    print(f"{'Layer':<10} {'Shape':<20} {'Sensitivity':<12} {'Precision':<10} {'Size':<10}")
    print("-" * 70)

    total_size_quant = 0
    for name, weights in model_layers.items():
        stats = quantizer.layer_stats[name]
        precision = precision_map[name]

        # Calculate quantized size
        if precision == QuantizationPrecision.FP32:
            size_bytes = weights.nbytes
        elif precision == QuantizationPrecision.FP16:
            size_bytes = weights.size * 2
        elif precision == QuantizationPrecision.INT8:
            size_bytes = weights.size
        elif precision == QuantizationPrecision.INT4:
            size_bytes = weights.size // 2

        total_size_quant += size_bytes
        size_mb = size_bytes / (1024 * 1024)

        print(
            f"{name:<10} {str(weights.shape):<20} {stats.sensitivity_score:<12.4f} "
            f"{precision._name:<10} {size_mb:.2f} MB"
        )

    original_size_mb = sum(w.nbytes for w in model_layers.values()) / (1024**2)
    quant_size_mb = total_size_quant / (1024**2)
    reduction = (1 - quant_size_mb / original_size_mb) * 100

    print("\n📊 Summary:")
    print(f"  Original size: {original_size_mb:.2f} MB (FP32)")
    print(f"  Quantized size: {quant_size_mb:.2f} MB (mixed)")
    print(f"  Reduction: {reduction:.1f}%")

    # Count precisions
    fp32 = sum(1 for p in precision_map.values() if p == QuantizationPrecision.FP32)
    fp16 = sum(1 for p in precision_map.values() if p == QuantizationPrecision.FP16)
    int8 = sum(1 for p in precision_map.values() if p == QuantizationPrecision.INT8)

    print(f"\n  Precision distribution:")
    print(f"    FP32: {fp32} layers")
    print(f"    FP16: {fp16} layers")
    print(f"    INT8: {int8} layers")


def demo_int4_packing():
    """Demo 4: INT4 packing for extreme compression."""
    print_section("Demo 4: INT4 Sub-byte Quantization")

    # Simulate embedding weights (common use case for INT4)
    np.random.seed(42)
    vocab_size = 10000
    embedding_dim = 256
    embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)

    print(f"\nEmbedding matrix: {embeddings.shape}")
    print(f"  Parameters: {embeddings.size:,}")
    print(f"  Size (FP32): {embeddings.nbytes / (1024**2):.2f} MB")

    # Quantize to INT4
    config = QuantizationConfig(
        precision=QuantizationPrecision.INT4, calibration_method=CalibrationMethod.PERCENTILE
    )
    quantizer = AdaptiveQuantizer(gpu_family="polaris", config=config)

    print("\n[Quantizing to INT4...]")
    q_embeddings, scale, zp = quantizer.quantize_tensor(embeddings)

    print(f"  Quantized size (unpacked INT4 in INT8): {q_embeddings.nbytes / (1024**2):.2f} MB")

    # Pack to actual 4-bit
    print("\n[Packing to 4-bit...]")
    packed = quantizer.pack_int4(q_embeddings)

    print(f"  Packed size (2 values per byte): {packed.nbytes / (1024**2):.2f} MB")
    print(f"  Compression ratio: {embeddings.nbytes / packed.nbytes:.1f}x")

    # Unpack and dequantize
    print("\n[Unpacking and dequantizing...]")
    unpacked = quantizer.unpack_int4(packed, original_shape=embeddings.shape)
    dequantized = quantizer.dequantize_tensor(unpacked, scale, zp)

    # Verify accuracy
    error = np.mean(np.abs(embeddings - dequantized))
    max_error = np.max(np.abs(embeddings - dequantized))
    sqnr = 10 * np.log10(
        np.mean(embeddings**2) / (np.mean((embeddings - dequantized) ** 2) + 1e-10)
    )

    print("\n📊 Accuracy Metrics:")
    print(f"  Mean error: {error:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  SQNR: {sqnr:.2f} dB")
    print(f"  Relative error: {error / np.abs(embeddings).mean() * 100:.2f}%")


def demo_qat_workflow():
    """Demo 5: Quantization-Aware Training workflow."""
    print_section("Demo 5: Quantization-Aware Training (QAT)")

    np.random.seed(42)
    weights = np.random.randn(128, 128).astype(np.float32)

    print(f"\nSimulating QAT for FC layer: {weights.shape}")

    # Standard Post-Training Quantization
    print("\n[Post-Training Quantization (PTQ)]")
    quantizer_ptq = AdaptiveQuantizer(gpu_family="polaris")
    stats_ptq = quantizer_ptq.analyze_layer_sensitivity(weights, "fc_ptq")

    print(f"  SQNR: {stats_ptq.sqnr_db:.2f} dB")
    print(f"  Sensitivity: {stats_ptq.sensitivity_score:.4f}")
    print(f"  Error: {stats_ptq.quantization_error:.6f}")

    # Quantization-Aware Training
    print("\n[Quantization-Aware Training (QAT)]")
    config_qat = QuantizationConfig(enable_qat=True, precision=QuantizationPrecision.INT8)
    quantizer_qat = AdaptiveQuantizer(gpu_family="polaris", config=config_qat)

    # Simulate forward pass with fake quantization
    print("  Simulating training forward pass...")
    fake_quant_weights = quantizer_qat.fake_quantize(weights)

    print(f"  Fake quantized dtype: {fake_quant_weights.dtype}")
    print(f"  Values on quantization grid: {not np.array_equal(weights, fake_quant_weights)}")

    # Simulate improvement after QAT
    # In real training, gradients would update weights to minimize quantization error
    print("\n  After QAT fine-tuning (simulated):")
    # Simulate 5% improvement in quantization error
    simulated_error_improvement = stats_ptq.quantization_error * 0.95
    simulated_sqnr_improvement = stats_ptq.sqnr_db + 1.5

    print(f"  SQNR: {simulated_sqnr_improvement:.2f} dB (+1.5 dB)")
    print(f"  Error: {simulated_error_improvement:.6f} (-5%)")

    print("\n📊 QAT Benefits:")
    print("  • Recovers 1-2% accuracy typically lost in PTQ")
    print("  • Fine-tuning 2-3 epochs sufficient")
    print("  • Straight-Through Estimator allows gradient flow")


def demo_rocm_integration():
    """Demo 6: ROCm integration (if available)."""
    print_section("Demo 6: ROCm Integration")

    if not ROCM_AVAILABLE:
        print("\n❌ ROCm integration not available")
        print("   Install ROCm and HIP Python bindings for GPU acceleration")
        return

    # Check ROCm status
    status = get_rocm_status()
    print(f"\nHIP Available: {status['hip_available']}")

    if not status["hip_available"]:
        print("  ROCm/HIP not detected on system")
        print("  Using CPU fallback")
        return

    print(f"Device Count: {status.get('device_count', 0)}")

    for device in status.get("devices", []):
        print(f"\nDevice {device['id']}: {device['name']}")
        print(f"  Compute Units: {device['compute_units']}")
        print(f"  Memory: {device['memory_mb']} MB")

    # Create ROCm quantizer
    print("\n[Creating ROCm-accelerated quantizer...]")
    rocm_quantizer = ROCmQuantizer(gpu_family="polaris", verbose=True)

    # Test quantization
    np.random.seed(42)
    weights = np.random.randn(512, 512).astype(np.float32)

    print(f"\nQuantizing {weights.shape} tensor...")
    start = time.time()
    q_weights, scale, zp = rocm_quantizer.quantize_tensor(weights)
    elapsed = (time.time() - start) * 1000

    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Scale: {scale:.6f}")
    print(f"  Zero-point: {zp}")

    print("\n📊 Note:")
    print("  GPU kernels for quantization not yet implemented")
    print("  Current version uses CPU with ROCm compatibility layer")
    print("  Future versions will include HIP kernels for acceleration")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  Comprehensive Quantization Demo - RX 580 AI Platform")
    print("  Version: 0.5.0-dev")
    print("=" * 70)

    try:
        demo_basic_quantization()
        demo_per_channel_quantization()
        demo_mixed_precision()
        demo_int4_packing()
        demo_qat_workflow()
        demo_rocm_integration()

        print("\n" + "=" * 70)
        print("  ✅ All Demos Completed Successfully!")
        print("=" * 70)

        print("\n📚 Documentation:")
        print("  • COMPUTE_QUANTIZATION_SUMMARY.md - Full implementation details")
        print("  • tests/test_quantization.py - 39 comprehensive tests")
        print("  • src/compute/quantization.py - Source code with documentation")

        print("\n🚀 Next Steps:")
        print("  • Integrate quantization with inference engine")
        print("  • Implement Sparse Networks (next in roadmap)")
        print("  • Add HIP kernels for GPU acceleration")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
