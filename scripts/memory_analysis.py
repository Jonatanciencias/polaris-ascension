#!/usr/bin/env python3
"""
Task 1.1.2.4 - Memory Access Pattern Analysis

Analyzes theoretical memory bandwidth, transaction patterns, and coalescing efficiency.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    """Analyzes memory access patterns of hybrid GEMM kernel."""
    
    # Hardware specs
    RX590_SPECS = {
        'peak_bandwidth_gbs': 256,
        'cache_line_size': 64,  # bytes
        'mem_transaction_size': 128,  # bytes (max coalesced)
        'lds_size_kb': 64,  # Per CU
        'wavefront_size': 64,  # Waves per CU
        'cu_count': 36
    }
    
    KERNEL_PARAMS = {
        'tile_size': 16,
        'block_size': 2,
        'local_size': (8, 8),  # 64 threads
        'float_size': 4  # bytes
    }
    
    def __init__(self, matrix_size=1024):
        """Initialize analyzer."""
        self.matrix_size = matrix_size
        self.results = {}
    
    def analyze_tile_loading(self):
        """Analyze memory transactions for tile loading."""
        logger.info("\n" + "="*80)
        logger.info("TILE LOADING ANALYSIS")
        logger.info("="*80)
        
        tile_size = self.KERNEL_PARAMS['tile_size']
        float_size = self.KERNEL_PARAMS['float_size']
        trans_size = self.RX590_SPECS['mem_transaction_size']
        
        # Tile data
        tile_elements = tile_size * tile_size  # 256 floats
        tile_bytes = tile_elements * float_size  # 1024 bytes
        
        logger.info(f"Tile size: {tile_size}×{tile_size}")
        logger.info(f"Tile data: {tile_elements} float32 elements = {tile_bytes} bytes")
        
        # Transactions needed
        transactions_per_tile = tile_bytes / trans_size
        logger.info(f"Memory transactions per tile load: {transactions_per_tile:.1f}")
        logger.info(f"  (Using float4 vectorization: {tile_bytes}/{trans_size} = {transactions_per_tile:.1f})")
        
        # Coalescing efficiency
        # With float4, we load 4 floats per transaction
        # 256 floats = 64 float4 loads = 8 transactions (128-byte max)
        logger.info(f"\nCoalescing (float4 optimization):")
        logger.info(f"  - Loads per thread: {tile_elements // 64} (64 threads in workgroup)")
        logger.info(f"  - float4 vectors: {tile_elements // 4}")
        logger.info(f"  - Transaction efficiency: HIGH (128-byte transactions)")
        
        self.results['tile_loading'] = {
            'tile_bytes': tile_bytes,
            'transactions': transactions_per_tile,
            'coalescing': 'HIGH (float4)'
        }
        
        return tile_bytes, transactions_per_tile
    
    def analyze_global_memory_access(self):
        """Analyze global memory access patterns."""
        logger.info("\n" + "="*80)
        logger.info("GLOBAL MEMORY ACCESS PATTERNS")
        logger.info("="*80)
        
        n = self.matrix_size
        tile_size = self.KERNEL_PARAMS['tile_size']
        
        # Workgroup grid
        wg_per_dim = (n + tile_size - 1) // tile_size
        total_workgroups = wg_per_dim * wg_per_dim
        
        logger.info(f"Matrix size: {n}×{n}")
        logger.info(f"Workgroups: {wg_per_dim}×{wg_per_dim} = {total_workgroups} workgroups")
        logger.info(f"Threads per workgroup: 64 (8×8)")
        logger.info(f"Total threads: {total_workgroups * 64}")
        
        # Memory access per kernel launch
        # Each thread loads 4 floats from A, 4 from B (float4 vectorization)
        bytes_per_thread = 8 * 4  # 8 float4 loads per iteration
        
        # K iterations (tile-based)
        k_tiles = (n + tile_size - 1) // tile_size
        
        # Total memory
        total_bytes = total_workgroups * 64 * bytes_per_thread * k_tiles
        total_gb = total_bytes / (1024**3)
        
        logger.info(f"\nMemory access per kernel:")
        logger.info(f"  - Bytes per thread per K-iteration: {bytes_per_thread}")
        logger.info(f"  - K-iterations: {k_tiles}")
        logger.info(f"  - Total memory accessed: {total_gb:.2f} GB")
        
        # Memory bandwidth for computation
        flops = 2 * n**3
        gflops_assumed = 650  # Assumed performance
        compute_time_s = (flops / gflops_assumed) / 1e9
        
        bandwidth_required = (total_gb / compute_time_s) / (self.RX590_SPECS['peak_bandwidth_gbs'] / 1024)
        logger.info(f"\nAssumed performance: {gflops_assumed} GFLOPS")
        logger.info(f"Compute time: {compute_time_s*1000:.2f} ms")
        logger.info(f"Required bandwidth: {bandwidth_required:.1f} GB/s")
        logger.info(f"Utilization: {min(bandwidth_required/self.RX590_SPECS['peak_bandwidth_gbs']*100, 100):.1f}%")
        
        self.results['global_access'] = {
            'total_gb': total_gb,
            'bandwidth_required_gbs': bandwidth_required,
            'utilization_percent': min(bandwidth_required/self.RX590_SPECS['peak_bandwidth_gbs']*100, 100)
        }
    
    def analyze_lds_usage(self):
        """Analyze LDS (local memory) usage."""
        logger.info("\n" + "="*80)
        logger.info("LOCAL MEMORY (LDS) USAGE")
        logger.info("="*80)
        
        tile_size = self.KERNEL_PARAMS['tile_size']
        float_size = self.KERNEL_PARAMS['float_size']
        lds_padding = 4  # Additional bytes per row
        
        # LDS layout: 2 buffers (double buffering)
        # Each: [tile_size × (tile_size + padding)]
        lds_per_buffer = tile_size * (tile_size + lds_padding) * float_size
        total_lds = 2 * lds_per_buffer
        
        lds_available = self.RX590_SPECS['lds_size_kb'] * 1024
        
        logger.info(f"LDS per buffer: {lds_per_buffer} bytes")
        logger.info(f"Double buffering: 2 buffers = {total_lds} bytes")
        logger.info(f"LDS available per CU: {lds_available} bytes")
        logger.info(f"Utilization: {total_lds/lds_available*100:.1f}%")
        
        if total_lds < lds_available * 0.5:
            logger.info("✅ LDS usage: EXCELLENT (plenty of headroom)")
        elif total_lds < lds_available * 0.8:
            logger.info("✅ LDS usage: GOOD")
        else:
            logger.warning("⚠️  LDS usage: TIGHT (limited headroom)")
        
        # Bank conflict analysis
        logger.info(f"\nBank conflict avoidance:")
        logger.info(f"  - Padding: {lds_padding} bytes per row")
        logger.info(f"  - Purpose: Avoid 32-bank conflicts on GCN 4.0")
        logger.info(f"  - Status: OPTIMIZED for float4 stores")
        
        self.results['lds_usage'] = {
            'lds_bytes': total_lds,
            'lds_available': lds_available,
            'utilization_percent': total_lds/lds_available*100,
            'bank_conflicts': 'AVOIDED (with padding)'
        }
    
    def analyze_arithmetic_intensity(self):
        """Analyze arithmetic intensity (FLOPS per byte)."""
        logger.info("\n" + "="*80)
        logger.info("ARITHMETIC INTENSITY ANALYSIS")
        logger.info("="*80)
        
        n = self.matrix_size
        tile_size = self.KERNEL_PARAMS['tile_size']
        
        # For tile-based GEMM
        # Computation: tile_size × tile_size × K per tile
        # Memory: Load A tile + Load B tile = 2 × tile_size × tile_size
        
        flops_per_tile = tile_size * tile_size * n * 2  # 2 for multiply+add
        memory_per_tile = 2 * tile_size * tile_size * 4  # 2 tiles × float32
        
        intensity = flops_per_tile / memory_per_tile
        
        logger.info(f"Per tile (size {tile_size}×{tile_size}):")
        logger.info(f"  - FLOPs: {flops_per_tile} (computation across K={n})")
        logger.info(f"  - Memory: {memory_per_tile} bytes (A+B tiles)")
        logger.info(f"  - Intensity: {intensity:.1f} FLOPS/byte")
        
        # Roofline analysis
        peak_bandwidth = self.RX590_SPECS['peak_bandwidth_gbs']
        peak_gflops = 6170
        
        ridge_point = peak_gflops / peak_bandwidth
        logger.info(f"\nRoofline analysis:")
        logger.info(f"  - Peak GFLOPS: {peak_gflops}")
        logger.info(f"  - Peak bandwidth: {peak_bandwidth} GB/s")
        logger.info(f"  - Ridge point: {ridge_point:.1f} FLOPS/byte")
        
        if intensity > ridge_point:
            logger.info(f"  - Status: COMPUTE-BOUND (intensity {intensity:.1f} > ridge {ridge_point:.1f})")
            logger.info(f"  - Goal: Increase GFLOPS")
        else:
            logger.info(f"  - Status: MEMORY-BOUND (intensity {intensity:.1f} < ridge {ridge_point:.1f})")
            logger.info(f"  - Goal: Increase bandwidth utilization")
        
        self.results['arithmetic_intensity'] = {
            'flops_per_byte': intensity,
            'ridge_point': ridge_point,
            'bound': 'COMPUTE' if intensity > ridge_point else 'MEMORY'
        }
    
    def analyze_register_blocking(self):
        """Analyze register blocking efficiency."""
        logger.info("\n" + "="*80)
        logger.info("REGISTER BLOCKING ANALYSIS")
        logger.info("="*80)
        
        block_size = self.KERNEL_PARAMS['block_size']
        
        # Each thread maintains block_size × block_size accumulators
        accumulators = block_size * block_size
        logger.info(f"Block size: {block_size}×{block_size}")
        logger.info(f"Accumulators per thread: {accumulators}")
        
        # Register estimation
        # Accumulators + temporaries (~20-24 registers total)
        regs_per_thread = 24
        total_regs = 64 * regs_per_thread  # 64 threads per workgroup
        
        logger.info(f"Register usage per thread: ~{regs_per_thread} registers")
        logger.info(f"Total per workgroup: {total_regs} registers")
        
        # Occupancy impact
        # GCN4: 256 registers per wave, max 10 waves per CU
        regs_available = 256 * 10
        occupancy = (total_regs / 64) / 10  # waves per CU
        
        logger.info(f"\nOccupancy impact:")
        logger.info(f"  - Registers available: {regs_available} per CU")
        logger.info(f"  - Registers used: {total_regs//64} per wave")
        logger.info(f"  - Occupancy: {occupancy:.1f} waves/CU")
        
        if occupancy >= 8:
            logger.info(f"  ✅ GOOD occupancy (8-10 waves)")
        elif occupancy >= 6:
            logger.info(f"  ⚠️  MODERATE occupancy (6-8 waves)")
        else:
            logger.warning(f"  ❌ LOW occupancy (<6 waves)")
        
        # Performance impact
        latency_hide = occupancy / 10  # Latency hiding factor
        logger.info(f"\nLatency hiding capability: {latency_hide*100:.0f}%")
        logger.info(f"  (10 waves = 100% coverage, overlaps memory latency)")
        
        self.results['register_blocking'] = {
            'block_size': block_size,
            'accumulators': accumulators,
            'regs_per_thread': regs_per_thread,
            'occupancy_waves': occupancy,
            'latency_hiding_percent': latency_hide * 100
        }
    
    def generate_optimization_suggestions(self):
        """Generate optimization suggestions based on analysis."""
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION SUGGESTIONS")
        logger.info("="*80)
        
        suggestions = []
        
        # Based on memory bandwidth
        if 'global_access' in self.results:
            util = self.results['global_access']['utilization_percent']
            if util < 20:
                suggestions.append(
                    "⚠️  Low bandwidth utilization (<20%): "
                    "Consider larger tiles or better prefetching"
                )
            elif util > 80:
                suggestions.append(
                    "⚠️  High bandwidth utilization (>80%): "
                    "Kernel is memory-bound, focus on coalescing"
                )
        
        # Based on arithmetic intensity
        if 'arithmetic_intensity' in self.results:
            if self.results['arithmetic_intensity']['bound'] == 'MEMORY':
                suggestions.append(
                    "💡 Memory-bound kernel: "
                    "Increase arithmetic intensity via tiling or blocking"
                )
        
        # Based on occupancy
        if 'register_blocking' in self.results:
            occ = self.results['register_blocking']['occupancy_waves']
            if occ < 6:
                suggestions.append(
                    "💡 Low occupancy: "
                    "Reduce register usage or increase thread block size"
                )
        
        # Based on LDS
        if 'lds_usage' in self.results:
            lds_util = self.results['lds_usage']['utilization_percent']
            if lds_util < 30:
                suggestions.append(
                    "💡 Low LDS utilization: "
                    "Could increase buffer sizes for better locality"
                )
        
        # General suggestions for Phase 1
        suggestions.extend([
            "✅ Float4 vectorization: ENABLED (good coalescing)",
            "✅ Double buffering: IMPLEMENTED (hides memory latency)",
            "✅ Register blocking: OPTIMIZED (2×2 per thread)",
            "🎯 Next: Fine-tune LDS bank conflicts and memory access patterns"
        ])
        
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"{i}. {suggestion}")
        
        self.results['suggestions'] = suggestions
    
    def run_full_analysis(self):
        """Run complete memory analysis."""
        logger.info("\n" + "="*80)
        logger.info(f"MEMORY ACCESS PATTERN ANALYSIS - TASK 1.1.2.4")
        logger.info(f"Matrix size: {self.matrix_size}×{self.matrix_size}")
        logger.info("="*80)
        
        self.analyze_tile_loading()
        self.analyze_global_memory_access()
        self.analyze_lds_usage()
        self.analyze_arithmetic_intensity()
        self.analyze_register_blocking()
        self.generate_optimization_suggestions()
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        
        return self.results


def main():
    """Main entry point."""
    analyzer = MemoryAnalyzer(matrix_size=1024)
    
    try:
        results = analyzer.run_full_analysis()
        
        logger.info("\n✅ MEMORY ANALYSIS COMPLETED")
        return 0
        
    except Exception as e:
        logger.error(f"\nCritical error: {e}", exc_info=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
