"""
GPU Performance Calculator - Core Layer
========================================

Mathematical models for GPU performance estimation based on hardware specifications.

Implements:
- Roofline model for TFLOPS calculation
- Memory bandwidth estimation
- GPU occupancy calculation
- Cache efficiency metrics

Based on GCN architecture specifications and academic research.

References:
- Williams et al. (2009): "Roofline: An Insightful Visual Performance Model"
- AMD GCN Architecture Whitepapers
- GPU Computing Gems (Hwu, 2011)

Version: 0.5.0-dev
License: MIT
"""

import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class VRAMType(Enum):
    """VRAM types with their characteristics"""
    GDDR5 = ("GDDR5", 2)  # (name, DDR multiplier)
    GDDR6 = ("GDDR6", 2)
    HBM = ("HBM", 2)
    HBM2 = ("HBM2", 2)
    
    @property
    def multiplier(self) -> int:
        return self.value[1]


@dataclass
class GPUSpecs:
    """Complete GPU hardware specifications"""
    compute_units: int
    clock_mhz: int
    boost_clock_mhz: int
    wavefront_size: int = 64
    max_wavefronts_per_cu: int = 40  # GCN architecture
    
    # Memory specs
    vram_gb: float = 0.0
    vram_type: VRAMType = VRAMType.GDDR5
    bus_width_bits: int = 256
    memory_clock_mhz: int = 2000
    
    # Cache hierarchy (KB)
    l1_cache_per_cu_kb: int = 16
    l2_cache_total_kb: int = 2048


class PerformanceCalculator:
    """
    Mathematical performance estimation for GCN GPUs.
    
    This class implements industry-standard models for:
    1. Peak TFLOPS calculation
    2. Memory bandwidth estimation
    3. GPU occupancy analysis
    4. Roofline model parameters
    """
    
    @staticmethod
    def calculate_theoretical_tflops(
        compute_units: int,
        clock_mhz: int,
        wavefront_size: int = 64,
        ops_per_cycle: int = 2,  # FMA = 2 ops
        use_boost: bool = True
    ) -> float:
        """
        Calculate theoretical peak TFLOPS for GCN architecture.
        
        Formula:
            TFLOPS = (CUs × Clock_MHz × Ops/cycle × Wavefront) / 10^6
        
        Derivation:
            - Each CU has 'wavefront_size' ALUs
            - Each ALU can do 'ops_per_cycle' operations per cycle (FMA)
            - At 'clock_mhz' MHz frequency
            - Total ops/sec = CUs × WF × ops/cycle × clock × 10^6
            - FLOPS = ops/sec, TFLOPS = ops/sec / 10^12
        
        Args:
            compute_units: Number of compute units (CUs)
            clock_mhz: Clock speed in MHz
            wavefront_size: Wavefront size (64 for GCN, 32 for RDNA)
            ops_per_cycle: Operations per cycle (2 for FMA)
            use_boost: Use boost clock if available
            
        Returns:
            Theoretical peak TFLOPS
            
        Example:
            For RX 580 (36 CUs, 1340 MHz boost):
            TFLOPS = (36 × 1340 × 2 × 64) / 10^6 = 6.17 TFLOPS
        """
        # Convert MHz to Hz, then to TFLOPS
        # ops/sec = CUs × WF × ops/cycle × MHz × 10^6
        # TFLOPS = ops/sec / 10^12 = (... × 10^6) / 10^12 = ... / 10^6
        
        ops_per_second = (
            compute_units * 
            wavefront_size * 
            ops_per_cycle * 
            clock_mhz
        )
        
        tflops = ops_per_second / 1e6
        return round(tflops, 2)
    
    @staticmethod
    def calculate_memory_bandwidth(
        bus_width_bits: int,
        memory_clock_mhz: int,
        vram_type: VRAMType = VRAMType.GDDR5
    ) -> float:
        """
        Calculate theoretical memory bandwidth.
        
        Formula:
            Bandwidth (GB/s) = (Bus_Width_bytes × Memory_Clock_MHz × DDR_multiplier) / 1000
        
        Derivation:
            - Bus width in bytes = bits / 8
            - DDR transfers data on both clock edges (multiplier = 2)
            - Effective clock = Memory_Clock × DDR_multiplier
            - Bandwidth = bytes_per_transfer × transfers_per_second
                        = (bus_width/8) × (clock × 2) × 10^6
                        = GB/s when divided by 10^9
        
        Args:
            bus_width_bits: Memory bus width (256 for RX 580)
            memory_clock_mhz: Memory clock speed
            vram_type: Type of VRAM (affects DDR multiplier)
            
        Returns:
            Theoretical bandwidth in GB/s
            
        Example:
            For RX 580 (256-bit, 2000 MHz GDDR5):
            BW = (256/8) × 2000 × 2 / 1000 = 256 GB/s
        """
        bus_width_bytes = bus_width_bits / 8
        ddr_multiplier = vram_type.multiplier
        
        # MB/s = bytes × MHz × multiplier
        bandwidth_mbs = bus_width_bytes * memory_clock_mhz * ddr_multiplier
        
        # Convert to GB/s
        bandwidth_gbs = bandwidth_mbs / 1000
        
        return round(bandwidth_gbs, 2)
    
    @staticmethod
    def calculate_occupancy(
        active_wavefronts: int,
        compute_units: int,
        max_wavefronts_per_cu: int = 40
    ) -> float:
        """
        Calculate GPU occupancy percentage.
        
        Formula:
            Occupancy = Active_Wavefronts / Max_Possible_Wavefronts
            Max = CUs × Max_WF_per_CU
        
        Occupancy directly affects:
        - Latency hiding (higher = better)
        - Resource utilization
        - Performance scaling
        
        Target: ≥ 75% for optimal performance
        
        Args:
            active_wavefronts: Currently executing wavefronts
            compute_units: Number of CUs
            max_wavefronts_per_cu: Maximum WFs per CU (40 for GCN)
            
        Returns:
            Occupancy as fraction [0.0, 1.0]
        """
        max_wavefronts = compute_units * max_wavefronts_per_cu
        occupancy = active_wavefronts / max_wavefronts
        
        return min(occupancy, 1.0)
    
    @staticmethod
    def calculate_arithmetic_intensity(
        flops: float,
        bytes_transferred: float
    ) -> float:
        """
        Calculate arithmetic intensity for roofline model.
        
        Formula:
            AI = FLOPS / Bytes_Transferred
        
        Interpretation:
            - High AI (>10): Compute-bound (good for GPU)
            - Low AI (<1): Memory-bound (bottleneck)
            - Medium AI (1-10): Balanced
        
        Args:
            flops: Floating point operations
            bytes_transferred: Data transferred to/from memory
            
        Returns:
            Arithmetic intensity in FLOPS/byte
        """
        if bytes_transferred == 0:
            return float('inf')
        
        return flops / bytes_transferred
    
    @staticmethod
    def roofline_performance(
        arithmetic_intensity: float,
        peak_tflops: float,
        memory_bandwidth_gbs: float
    ) -> Tuple[float, str]:
        """
        Apply roofline model to predict actual performance.
        
        Formula:
            Perf = min(Peak_TFLOPS, AI × Memory_BW)
        
        The roofline model determines whether a workload is:
        - Compute-bound: Limited by TFLOPS
        - Memory-bound: Limited by bandwidth
        
        Args:
            arithmetic_intensity: FLOPS per byte
            peak_tflops: Theoretical peak TFLOPS
            memory_bandwidth_gbs: Memory bandwidth in GB/s
            
        Returns:
            (Achievable TFLOPS, "compute-bound" | "memory-bound")
        """
        # Memory-bound ceiling: AI × BW (FLOPS/byte × GB/s = GFLOPS)
        # Convert GB/s to TB/s for TFLOPS: GB/s / 1000
        memory_ceiling = arithmetic_intensity * (memory_bandwidth_gbs / 1000)
        
        achievable = min(peak_tflops, memory_ceiling)
        bound = "compute-bound" if achievable == peak_tflops else "memory-bound"
        
        return round(achievable, 2), bound
    
    @staticmethod
    def optimal_batch_size(
        model_size_mb: float,
        available_vram_mb: float,
        activation_ratio: float = 0.3,
        gradient_ratio: float = 0.2,
        overhead_ratio: float = 0.1
    ) -> int:
        """
        Calculate optimal batch size using memory constraints.
        
        Formula:
            Per-sample memory = Model × (1 + activation + gradient + overhead)
            Batch_Size = floor(Available / Per_Sample)
        
        Memory breakdown per sample:
        - Model weights: Fixed
        - Activations: ~30% of model (layer outputs)
        - Gradients: ~20% of model (backprop)
        - Overhead: ~10% (framework overhead)
        
        Args:
            model_size_mb: Model size in MB
            available_vram_mb: Available VRAM
            activation_ratio: Activation memory ratio
            gradient_ratio: Gradient memory ratio  
            overhead_ratio: Framework overhead ratio
            
        Returns:
            Optimal batch size (≥1)
        """
        per_sample_mb = model_size_mb * (
            1.0 +  # Model itself (shared)
            activation_ratio +
            gradient_ratio +
            overhead_ratio
        )
        
        # Reserve model memory (loaded once)
        available_for_batch = available_vram_mb - model_size_mb
        
        if available_for_batch <= 0:
            return 1
        
        # Calculate batch size
        batch_size = int(available_for_batch / per_sample_mb)
        
        return max(1, batch_size)
    
    @classmethod
    def analyze_gpu(cls, specs: GPUSpecs) -> Dict[str, any]:
        """
        Comprehensive GPU performance analysis.
        
        Returns dictionary with:
        - Theoretical maximums
        - Bottleneck analysis
        - Optimization recommendations
        
        Args:
            specs: Complete GPU specifications
            
        Returns:
            Performance analysis dictionary
        """
        # Calculate peak performance
        peak_tflops = cls.calculate_theoretical_tflops(
            specs.compute_units,
            specs.boost_clock_mhz,
            specs.wavefront_size
        )
        
        bandwidth = cls.calculate_memory_bandwidth(
            specs.bus_width_bits,
            specs.memory_clock_mhz,
            specs.vram_type
        )
        
        # Calculate cache sizes
        total_l1_kb = specs.compute_units * specs.l1_cache_per_cu_kb
        
        # Estimate practical performance (85% of peak due to overheads)
        practical_tflops = peak_tflops * 0.85
        
        # Calculate compute-to-bandwidth ratio
        # TFLOPS / (GB/s) = operations per byte
        # Higher = more compute per byte (better for heavy compute)
        compute_intensity = (peak_tflops * 1000) / bandwidth  # GFLOPS per GB/s
        
        return {
            'peak_tflops': peak_tflops,
            'practical_tflops': practical_tflops,
            'memory_bandwidth_gbs': bandwidth,
            'compute_intensity_ratio': round(compute_intensity, 2),
            'total_l1_cache_kb': total_l1_kb,
            'total_l2_cache_kb': specs.l2_cache_total_kb,
            'max_occupancy_wavefronts': specs.compute_units * specs.max_wavefronts_per_cu,
            'recommendation': cls._get_recommendation(compute_intensity)
        }
    
    @staticmethod
    def _get_recommendation(compute_intensity: float) -> str:
        """Get optimization recommendation based on compute intensity."""
        if compute_intensity > 30:
            return "Excellent for compute-heavy workloads (convolutions, GEMM)"
        elif compute_intensity > 20:
            return "Good balance for mixed workloads"
        elif compute_intensity > 10:
            return "Consider memory-efficient algorithms (sparse ops)"
        else:
            return "Memory-bound: Prioritize data reuse and caching"


# Polaris GPU specifications database
POLARIS_SPECS = {
    "RX 580": GPUSpecs(
        compute_units=36,
        clock_mhz=1120,
        boost_clock_mhz=1340,
        vram_gb=8.0,
        vram_type=VRAMType.GDDR5,
        bus_width_bits=256,
        memory_clock_mhz=2000,
        l1_cache_per_cu_kb=16,
        l2_cache_total_kb=2048
    ),
    "RX 570": GPUSpecs(
        compute_units=32,
        clock_mhz=1168,
        boost_clock_mhz=1244,
        vram_gb=4.0,
        bus_width_bits=256,
        memory_clock_mhz=1750,
        l1_cache_per_cu_kb=16,
        l2_cache_total_kb=2048
    ),
    "RX 480": GPUSpecs(
        compute_units=36,
        clock_mhz=1120,
        boost_clock_mhz=1266,
        vram_gb=8.0,
        bus_width_bits=256,
        memory_clock_mhz=2000,
        l1_cache_per_cu_kb=16,
        l2_cache_total_kb=2048
    ),
}


if __name__ == "__main__":
    # Demo: Analyze RX 580
    print("=" * 60)
    print("GPU Performance Analysis Demo")
    print("=" * 60)
    
    rx580 = POLARIS_SPECS["RX 580"]
    analysis = PerformanceCalculator.analyze_gpu(rx580)
    
    print(f"\nRX 580 Analysis:")
    print(f"  Peak TFLOPS: {analysis['peak_tflops']}")
    print(f"  Practical TFLOPS: {analysis['practical_tflops']}")
    print(f"  Memory Bandwidth: {analysis['memory_bandwidth_gbs']} GB/s")
    print(f"  Compute Intensity: {analysis['compute_intensity_ratio']}")
    print(f"  L1 Cache: {analysis['total_l1_cache_kb']} KB")
    print(f"  L2 Cache: {analysis['total_l2_cache_kb']} KB")
    print(f"  Max Wavefronts: {analysis['max_occupancy_wavefronts']}")
    print(f"  Recommendation: {analysis['recommendation']}")
    
    # Test batch size calculation
    print(f"\n{'='*60}")
    print("Batch Size Optimization")
    print("=" * 60)
    
    test_models = [512, 2048, 4096]
    available = 7168  # 7GB available (1GB headroom)
    
    for model_mb in test_models:
        batch_size = PerformanceCalculator.optimal_batch_size(model_mb, available)
        print(f"  Model {model_mb}MB → Batch size: {batch_size}")
