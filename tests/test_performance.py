"""
Tests for Performance Calculator
"""
import pytest
from core.performance import (
    PerformanceCalculator,
    GPUSpecs,
    VRAMType,
    POLARIS_SPECS
)


def test_tflops_calculation():
    """Test theoretical TFLOPS calculation"""
    # RX 580: 36 CUs × 1340 MHz × 2 ops/cycle × 64 WF / 10^6 = 6.17 TFLOPS
    tflops = PerformanceCalculator.calculate_theoretical_tflops(
        compute_units=36,
        clock_mhz=1340,
        wavefront_size=64,
        ops_per_cycle=2
    )
    
    assert tflops == 6.17
    
    
def test_memory_bandwidth_calculation():
    """Test memory bandwidth calculation"""
    # RX 580: (256/8) × 2000 × 2 / 1000 = 128 GB/s
    bandwidth = PerformanceCalculator.calculate_memory_bandwidth(
        bus_width_bits=256,
        memory_clock_mhz=2000,
        vram_type=VRAMType.GDDR5
    )
    
    assert bandwidth == 128.0


def test_occupancy_calculation():
    """Test GPU occupancy calculation"""
    # 720 active WFs / (36 CUs × 40 WFs/CU) = 0.5 = 50%
    occupancy = PerformanceCalculator.calculate_occupancy(
        active_wavefronts=720,
        compute_units=36,
        max_wavefronts_per_cu=40
    )
    
    assert occupancy == 0.5
    
    # Test capping at 100%
    occupancy = PerformanceCalculator.calculate_occupancy(
        active_wavefronts=2000,
        compute_units=36
    )
    
    assert occupancy == 1.0


def test_arithmetic_intensity():
    """Test arithmetic intensity calculation"""
    # 1000 FLOPS / 100 bytes = 10 FLOPS/byte
    ai = PerformanceCalculator.calculate_arithmetic_intensity(
        flops=1000,
        bytes_transferred=100
    )
    
    assert ai == 10.0
    
    # Test divide by zero
    ai = PerformanceCalculator.calculate_arithmetic_intensity(
        flops=1000,
        bytes_transferred=0
    )
    
    assert ai == float('inf')


def test_roofline_model():
    """Test roofline model performance prediction"""
    # Test compute-bound
    perf, bound = PerformanceCalculator.roofline_performance(
        arithmetic_intensity=100,  # High AI
        peak_tflops=6.17,
        memory_bandwidth_gbs=128
    )
    
    assert bound == "compute-bound"
    assert perf == 6.17
    
    # Test memory-bound
    perf, bound = PerformanceCalculator.roofline_performance(
        arithmetic_intensity=0.01,  # Low AI
        peak_tflops=6.17,
        memory_bandwidth_gbs=128
    )
    
    assert bound == "memory-bound"
    assert perf < 6.17


def test_optimal_batch_size():
    """Test batch size calculation"""
    # Model: 2048 MB, Available: 8192 MB
    # Per sample: 2048 × (1.0 + 0.3 + 0.2 + 0.1) = 3276.8 MB
    # Available for batch: 8192 - 2048 = 6144 MB
    # Batch size: floor(6144 / 3276.8) = 1
    
    batch_size = PerformanceCalculator.optimal_batch_size(
        model_size_mb=2048,
        available_vram_mb=8192
    )
    
    assert batch_size >= 1
    
    # Test with larger available memory
    batch_size = PerformanceCalculator.optimal_batch_size(
        model_size_mb=512,
        available_vram_mb=8192
    )
    
    assert batch_size > 1


def test_analyze_gpu():
    """Test comprehensive GPU analysis"""
    rx580 = POLARIS_SPECS["RX 580"]
    analysis = PerformanceCalculator.analyze_gpu(rx580)
    
    assert "peak_tflops" in analysis
    assert "practical_tflops" in analysis
    assert "memory_bandwidth_gbs" in analysis
    assert "compute_intensity_ratio" in analysis
    assert "recommendation" in analysis
    
    # Verify values
    assert analysis["peak_tflops"] == 6.17
    assert analysis["memory_bandwidth_gbs"] == 128.0
    assert 0 < analysis["practical_tflops"] < analysis["peak_tflops"]


def test_polaris_specs_database():
    """Test Polaris specs database completeness"""
    assert "RX 580" in POLARIS_SPECS
    assert "RX 570" in POLARIS_SPECS
    assert "RX 480" in POLARIS_SPECS
    
    rx580 = POLARIS_SPECS["RX 580"]
    assert rx580.compute_units == 36
    assert rx580.vram_gb == 8.0
    assert rx580.bus_width_bits == 256


def test_edge_cases():
    """Test edge cases and error handling"""
    # Zero compute units
    tflops = PerformanceCalculator.calculate_theoretical_tflops(
        compute_units=0,
        clock_mhz=1000
    )
    assert tflops == 0.0
    
    # Zero clock
    tflops = PerformanceCalculator.calculate_theoretical_tflops(
        compute_units=36,
        clock_mhz=0
    )
    assert tflops == 0.0
    
    # Very small model
    batch_size = PerformanceCalculator.optimal_batch_size(
        model_size_mb=1,
        available_vram_mb=8192
    )
    assert batch_size > 100  # Should allow many samples
