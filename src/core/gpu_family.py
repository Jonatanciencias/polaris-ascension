"""
GPU Family Detection and Abstraction
====================================

This module provides GPU family support for the Legacy GPU AI Platform,
with primary focus on AMD Polaris (GCN 4.0) architecture.

SUPPORTED GPU Families (Tested):
-------------------------------
1. Polaris (GCN 4.0) - PRIMARY & TESTED
   - RX 580, RX 570, RX 480, RX 470
   - RX 560, RX 550 (limited)
   - 8GB / 4GB variants
   - Compute Units: 16-36
   - This is the ONLY architecture we can test directly

COMMUNITY-CONTRIBUTED (Untested):
--------------------------------
2. Vega (GCN 5.0) - Community contributions welcome
   - Vega 56, Vega 64
   - Should work but NOT TESTED by maintainers
   
3. Other GCN - May work with degraded performance
   - GCN 3.0 (Tonga, Fiji)
   - GCN 2.0 (Bonaire, Hawaii)

NOTE: RDNA (Navi) architecture has different wavefront size (32 vs 64)
and may require different optimizations. Not officially supported.

Architecture Abstraction:
------------------------
This module abstracts away GPU-specific differences so that the
upper layers (SDK, Inference) can work without knowing the specifics.

Example:
-------
    from src.core.gpu_family import GPUFamily, detect_gpu_family
    
    family = detect_gpu_family()
    print(f"Detected: {family.name} ({family.architecture})")
    print(f"Recommended batch size: {family.recommended_batch_size}")

Version: 0.5.0-dev
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import subprocess
import re


class Architecture(Enum):
    """AMD GPU architectures."""
    GCN_1_0 = "gcn1.0"     # Southern Islands (HD 7000)
    GCN_2_0 = "gcn2.0"     # Sea Islands (R7/R9 200)
    GCN_3_0 = "gcn3.0"     # Volcanic Islands (R9 285/380)
    GCN_4_0 = "gcn4.0"     # Polaris - PRIMARY TARGET
    GCN_5_0 = "gcn5.0"     # Vega - Community supported
    RDNA_1_0 = "rdna1.0"   # Navi - NOT SUPPORTED (different wavefront)
    RDNA_2_0 = "rdna2.0"   # Big Navi - NOT SUPPORTED
    UNKNOWN = "unknown"


class SupportLevel(Enum):
    """Level of support for a GPU family."""
    TESTED = "tested"           # Tested by maintainers
    COMMUNITY = "community"     # Community contributed, untested
    EXPERIMENTAL = "experimental"  # May work, no guarantees
    UNSUPPORTED = "unsupported"    # Known incompatible


@dataclass
class GPUCapabilities:
    """Hardware capabilities of a GPU family."""
    fp32_tflops: float
    fp16_tflops: float
    memory_bandwidth_gbps: float
    has_rapid_packed_math: bool = False
    has_tensor_cores: bool = False
    wavefront_size: int = 64
    max_workgroup_size: int = 256
    local_memory_kb: int = 64


@dataclass
class GPUFamily:
    """
    Represents a GPU family with its characteristics.
    
    This class encapsulates all the information needed to optimize
    code for a specific GPU family.
    """
    name: str
    architecture: Architecture
    compute_units: int
    vram_gb: float
    capabilities: GPUCapabilities
    
    # Optimization hints
    recommended_batch_size: int = 4
    recommended_precision: str = "fp32"
    memory_strategy: str = "conservative"
    
    # Identification patterns
    device_patterns: List[str] = field(default_factory=list)
    
    def get_optimization_profile(self) -> dict:
        """
        Get optimization profile for this GPU family.
        
        Returns:
            dict with optimization settings
        """
        return {
            "batch_size": self.recommended_batch_size,
            "precision": self.recommended_precision,
            "memory_strategy": self.memory_strategy,
            "wavefront_size": self.capabilities.wavefront_size,
            "use_fp16": self.capabilities.has_rapid_packed_math,
        }
    
    def estimate_model_fit(self, model_size_mb: float) -> dict:
        """
        Estimate if a model will fit in VRAM.
        
        Args:
            model_size_mb: Model size in megabytes
            
        Returns:
            dict with fit analysis
        """
        vram_mb = self.vram_gb * 1024
        
        # Account for framework overhead (~500MB)
        available_mb = vram_mb - 500
        
        # Estimate activations (roughly 2x model size for batch=1)
        estimated_activations = model_size_mb * 2 * self.recommended_batch_size
        
        total_required = model_size_mb + estimated_activations
        
        fits = total_required < available_mb
        max_batch = int(available_mb / (model_size_mb * 3)) if model_size_mb > 0 else 0
        
        return {
            "fits": fits,
            "model_size_mb": model_size_mb,
            "available_vram_mb": available_mb,
            "estimated_total_mb": total_required,
            "max_batch_size": max_batch,
            "recommendation": "OK" if fits else "Use smaller model or reduce batch size",
        }


# Pre-defined GPU family configurations

POLARIS_8GB = GPUFamily(
    name="Polaris 8GB",
    architecture=Architecture.GCN_4_0,
    compute_units=36,
    vram_gb=8.0,
    capabilities=GPUCapabilities(
        fp32_tflops=6.17,
        fp16_tflops=6.17,  # No FP16 acceleration
        memory_bandwidth_gbps=256,
        has_rapid_packed_math=False,
        wavefront_size=64,
    ),
    recommended_batch_size=4,
    recommended_precision="fp32",
    memory_strategy="conservative",
    device_patterns=["RX 580", "RX 480", "Polaris 20"],
)

POLARIS_4GB = GPUFamily(
    name="Polaris 4GB",
    architecture=Architecture.GCN_4_0,
    compute_units=32,
    vram_gb=4.0,
    capabilities=GPUCapabilities(
        fp32_tflops=5.1,
        fp16_tflops=5.1,
        memory_bandwidth_gbps=224,
        has_rapid_packed_math=False,
        wavefront_size=64,
    ),
    recommended_batch_size=2,
    recommended_precision="fp32",
    memory_strategy="aggressive",
    device_patterns=["RX 570", "RX 470", "Polaris 10"],
)

VEGA_64 = GPUFamily(
    name="Vega 64",
    architecture=Architecture.GCN_5_0,
    compute_units=64,
    vram_gb=8.0,
    capabilities=GPUCapabilities(
        fp32_tflops=12.66,
        fp16_tflops=25.32,  # Rapid Packed Math!
        memory_bandwidth_gbps=484,
        has_rapid_packed_math=True,
        wavefront_size=64,
    ),
    recommended_batch_size=8,
    recommended_precision="fp16",
    memory_strategy="moderate",
    device_patterns=["Vega 64", "Radeon RX Vega 64"],
)

VEGA_56 = GPUFamily(
    name="Vega 56",
    architecture=Architecture.GCN_5_0,
    compute_units=56,
    vram_gb=8.0,
    capabilities=GPUCapabilities(
        fp32_tflops=10.5,
        fp16_tflops=21.0,
        memory_bandwidth_gbps=410,
        has_rapid_packed_math=True,
        wavefront_size=64,
    ),
    recommended_batch_size=8,
    recommended_precision="fp16",
    memory_strategy="moderate",
    device_patterns=["Vega 56", "Radeon RX Vega 56"],
)

NAVI_5700XT = GPUFamily(
    name="Navi 5700 XT (UNSUPPORTED)",
    architecture=Architecture.RDNA_1_0,
    compute_units=40,
    vram_gb=8.0,
    capabilities=GPUCapabilities(
        fp32_tflops=9.75,
        fp16_tflops=19.5,
        memory_bandwidth_gbps=448,
        has_rapid_packed_math=True,
        wavefront_size=32,  # RDNA uses Wave32 - INCOMPATIBLE with GCN optimizations!
    ),
    recommended_batch_size=8,
    recommended_precision="fp16",
    memory_strategy="moderate",
    device_patterns=["5700 XT", "Navi 10"],
)

# Additional Polaris variants
POLARIS_LITE = GPUFamily(
    name="Polaris Lite (RX 560/550)",
    architecture=Architecture.GCN_4_0,
    compute_units=16,
    vram_gb=4.0,
    capabilities=GPUCapabilities(
        fp32_tflops=2.6,
        fp16_tflops=2.6,
        memory_bandwidth_gbps=112,
        has_rapid_packed_math=False,
        wavefront_size=64,
    ),
    recommended_batch_size=1,
    recommended_precision="fp32",
    memory_strategy="aggressive",
    device_patterns=["RX 560", "RX 550", "Polaris 11", "Polaris 12"],
)

# GPU family registry with support levels
GPU_FAMILIES: Dict[str, GPUFamily] = {
    # TESTED - Primary support
    "polaris_8gb": POLARIS_8GB,
    "polaris_4gb": POLARIS_4GB,
    "polaris_lite": POLARIS_LITE,
    # COMMUNITY - May work, untested
    "vega_64": VEGA_64,
    "vega_56": VEGA_56,
    # UNSUPPORTED - Different architecture
    "navi_5700xt": NAVI_5700XT,
}

# Support level mapping
GPU_SUPPORT_LEVELS: Dict[str, SupportLevel] = {
    "polaris_8gb": SupportLevel.TESTED,
    "polaris_4gb": SupportLevel.TESTED,
    "polaris_lite": SupportLevel.TESTED,
    "vega_64": SupportLevel.COMMUNITY,
    "vega_56": SupportLevel.COMMUNITY,
    "navi_5700xt": SupportLevel.UNSUPPORTED,
}


def get_support_level(family_id: str) -> SupportLevel:
    """Get the support level for a GPU family."""
    return GPU_SUPPORT_LEVELS.get(family_id.lower(), SupportLevel.EXPERIMENTAL)


def detect_gpu_family() -> GPUFamily:
    """
    Auto-detect the GPU family from system hardware.
    
    Returns:
        Detected GPUFamily or default (Polaris 8GB)
        
    Note:
        - Polaris GPUs are fully supported and tested
        - Vega GPUs may work but are community-supported
        - RDNA (Navi) GPUs are NOT supported due to architecture differences
    """
    device_name = _get_gpu_device_name()
    
    if not device_name:
        print("Warning: Could not detect GPU. Using default Polaris 8GB profile.")
        return POLARIS_8GB
    
    device_lower = device_name.lower()
    
    # Check for unsupported RDNA first and warn
    if any(x in device_lower for x in ["5700", "5600", "5500", "navi", "6700", "6800", "6900"]):
        print("⚠️  WARNING: RDNA (Navi) GPU detected!")
        print("    This architecture uses Wave32 instead of Wave64.")
        print("    The platform is optimized for GCN (Polaris/Vega).")
        print("    Proceeding with limited compatibility mode...")
        return NAVI_5700XT
    
    # Check against known Polaris patterns (TESTED)
    if "580" in device_lower or "480" in device_lower:
        print("✅ Detected: Polaris 8GB (RX 580/480) - TESTED & SUPPORTED")
        return POLARIS_8GB
    elif "570" in device_lower or "470" in device_lower:
        print("✅ Detected: Polaris 4GB (RX 570/470) - TESTED & SUPPORTED")
        return POLARIS_4GB
    elif "560" in device_lower or "550" in device_lower:
        print("✅ Detected: Polaris Lite (RX 560/550) - TESTED & SUPPORTED")
        return POLARIS_LITE
        
    # Vega (community supported)
    elif "vega 64" in device_lower or "vega64" in device_lower:
        print("⚡ Detected: Vega 64 - COMMUNITY SUPPORTED (untested by maintainers)")
        return VEGA_64
    elif "vega 56" in device_lower or "vega56" in device_lower:
        print("⚡ Detected: Vega 56 - COMMUNITY SUPPORTED (untested by maintainers)")
        return VEGA_56
    
    # Check generic patterns
    for family_id, family in GPU_FAMILIES.items():
        for pattern in family.device_patterns:
            if pattern.lower() in device_lower:
                support = get_support_level(family_id)
                print(f"Detected GPU family: {family.name} ({support.value})")
                return family
    
    print(f"Warning: Unknown GPU '{device_name}'. Using default Polaris 8GB profile.")
    print("  If you have a Polaris/Vega GPU, please report this for better detection.")
    return POLARIS_8GB


def _get_gpu_device_name() -> Optional[str]:
    """Get GPU device name from system."""
    try:
        # Try rocm-smi first
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Parse output to find device name
            for line in result.stdout.split('\n'):
                if 'Card series' in line or 'GPU' in line:
                    match = re.search(r':\s*(.+)$', line)
                    if match:
                        return match.group(1).strip()
                        
    except Exception:
        pass
        
    try:
        # Fallback to lspci
        result = subprocess.run(
            ["lspci", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'VGA' in line and 'AMD' in line:
                    # Extract GPU name
                    match = re.search(r'\[(.+)\]', line)
                    if match:
                        return match.group(1)
                        
    except Exception:
        pass
    
    return None


def get_family_by_id(family_id: str) -> Optional[GPUFamily]:
    """
    Get GPU family by identifier.
    
    Args:
        family_id: Family identifier (e.g., "polaris_8gb")
        
    Returns:
        GPUFamily or None
    """
    return GPU_FAMILIES.get(family_id.lower())


def list_supported_families() -> List[str]:
    """
    List all supported GPU families.
    
    Returns:
        List of family identifiers
    """
    return list(GPU_FAMILIES.keys())


def compare_families(family_a: str, family_b: str) -> dict:
    """
    Compare two GPU families.
    
    Args:
        family_a: First family identifier
        family_b: Second family identifier
        
    Returns:
        Comparison dict
    """
    a = GPU_FAMILIES.get(family_a.lower())
    b = GPU_FAMILIES.get(family_b.lower())
    
    if not a or not b:
        return {"error": "Unknown family identifier"}
    
    return {
        "families": [a.name, b.name],
        "comparison": {
            "compute_units": {a.name: a.compute_units, b.name: b.compute_units},
            "vram_gb": {a.name: a.vram_gb, b.name: b.vram_gb},
            "fp32_tflops": {
                a.name: a.capabilities.fp32_tflops,
                b.name: b.capabilities.fp32_tflops
            },
            "fp16_tflops": {
                a.name: a.capabilities.fp16_tflops,
                b.name: b.capabilities.fp16_tflops
            },
            "recommended_batch": {
                a.name: a.recommended_batch_size,
                b.name: b.recommended_batch_size
            },
            "has_rapid_packed_math": {
                a.name: a.capabilities.has_rapid_packed_math,
                b.name: b.capabilities.has_rapid_packed_math
            },
        },
        "winner_by_metric": {
            "compute_units": a.name if a.compute_units > b.compute_units else b.name,
            "fp32_performance": a.name if a.capabilities.fp32_tflops > b.capabilities.fp32_tflops else b.name,
            "fp16_performance": a.name if a.capabilities.fp16_tflops > b.capabilities.fp16_tflops else b.name,
        }
    }


def print_family_info(family: GPUFamily):
    """Print formatted information about a GPU family."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║ GPU Family: {family.name:<52} ║
╠══════════════════════════════════════════════════════════════════╣
║ Architecture: {family.architecture.value:<50} ║
║ Compute Units: {family.compute_units:<49} ║
║ VRAM: {family.vram_gb:.1f} GB{' '*54} ║
╠══════════════════════════════════════════════════════════════════╣
║ Performance:                                                     ║
║   • FP32: {family.capabilities.fp32_tflops:.2f} TFLOPS{' '*45} ║
║   • FP16: {family.capabilities.fp16_tflops:.2f} TFLOPS{' '*45} ║
║   • Memory BW: {family.capabilities.memory_bandwidth_gbps:.0f} GB/s{' '*40} ║
╠══════════════════════════════════════════════════════════════════╣
║ Optimizations:                                                   ║
║   • Recommended Batch: {family.recommended_batch_size:<41} ║
║   • Precision: {family.recommended_precision:<50} ║
║   • Memory Strategy: {family.memory_strategy:<43} ║
║   • Rapid Packed Math: {'Yes' if family.capabilities.has_rapid_packed_math else 'No':<42} ║
╚══════════════════════════════════════════════════════════════════╝
""")


# Self-test
if __name__ == "__main__":
    print("GPU Family Detection Test")
    print("=" * 60)
    
    detected = detect_gpu_family()
    print_family_info(detected)
    
    print("\nSupported families:", list_supported_families())
    
    print("\nComparing Polaris 8GB vs Vega 64:")
    comparison = compare_families("polaris_8gb", "vega_64")
    print(f"  FP32 winner: {comparison['winner_by_metric']['fp32_performance']}")
    print(f"  FP16 winner: {comparison['winner_by_metric']['fp16_performance']}")
