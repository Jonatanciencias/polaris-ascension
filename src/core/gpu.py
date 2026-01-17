"""
GPU Management Module - Core Layer
==================================

Professional GPU detection, initialization, and resource management
specifically optimized for AMD GCN architecture (Polaris family).

This module provides:
- Automatic GPU family detection
- OpenCL platform initialization with GCN optimizations
- Hardware capability queries
- Compute backend selection

Supported Hardware:
- AMD Polaris (RX 400/500 series) - Primary
- AMD Vega (Community supported)

Version: 0.5.0-dev
License: MIT
"""

import subprocess
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import os


@dataclass
class GPUInfo:
    """Comprehensive GPU information container"""
    # Basic identification
    device_name: str
    pci_id: str
    architecture: str
    family: str = "unknown"
    
    # Memory
    memory_total_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    
    # Compute capabilities
    compute_units: int = 0
    wavefront_size: int = 64  # GCN default
    max_clock_mhz: int = 0
    
    # Backend availability
    driver_version: Optional[str] = None
    opencl_available: bool = False
    opencl_version: Optional[str] = None
    rocm_available: bool = False
    rocm_version: Optional[str] = None
    
    # Performance hints
    fp32_tflops: float = 0.0
    recommended_batch_size: int = 1


class GPUDetectionError(Exception):
    """Raised when GPU detection fails"""
    pass


class GPUManager:
    """
    Professional GPU resource manager for AMD Polaris architecture.
    
    This class handles all GPU-related operations including:
    - Hardware detection and identification
    - Family classification (Polaris, Vega, etc.)
    - Backend (OpenCL/ROCm) initialization
    - Performance profiling and recommendations
    
    Example:
        manager = GPUManager()
        if manager.initialize():
            info = manager.get_info()
            print(f"Detected: {info['device_name']}")
            print(f"Compute units: {info['compute_units']}")
    """
    
    def __init__(self, auto_detect: bool = True):
        """
        Initialize GPU Manager.
        
        Args:
            auto_detect: Automatically detect GPU on initialization
        """
        self._gpu_info: Optional[GPUInfo] = None
        self._initialized = False
        self._opencl_context = None
        self._gpu_family = None
        
        if auto_detect:
            try:
                self.detect_gpu()
            except GPUDetectionError as e:
                print(f"Warning: {e}")
    
    def detect_gpu(self) -> GPUInfo:
        """
        Detect AMD GPU in the system and classify it.
        
        Returns:
            GPUInfo object with complete hardware information
            
        Raises:
            GPUDetectionError: If no compatible GPU is found
        """
        try:
            # Method 1: Try lspci (most reliable on Linux)
            gpu_info = self._detect_via_lspci()
            
            if not gpu_info:
                # Method 2: Try ROCm SMI
                gpu_info = self._detect_via_rocm()
            
            if not gpu_info:
                # Method 3: Try OpenCL
                gpu_info = self._detect_via_opencl()
            
            if not gpu_info:
                raise GPUDetectionError("No AMD GPU detected. Supported: Polaris (RX 400/500)")
            
            # Enrich with family information
            self._classify_gpu_family(gpu_info)
            
            # Check backend availability
            gpu_info.opencl_available, gpu_info.opencl_version = self._check_opencl()
            gpu_info.rocm_available, gpu_info.rocm_version = self._check_rocm()
            
            # Get VRAM info
            gpu_info.memory_total_gb = self._get_vram_gb()
            
            # Calculate performance metrics
            self._calculate_performance_metrics(gpu_info)
            
            self._gpu_info = gpu_info
            return gpu_info
            
        except Exception as e:
            raise GPUDetectionError(f"GPU detection failed: {e}")
    
    def _detect_via_lspci(self) -> Optional[GPUInfo]:
        """Detect GPU using lspci command."""
        try:
            result = subprocess.run(
                ["lspci", "-v"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            
            lines = result.stdout.split('\n')
            
            for i, line in enumerate(lines):
                # Look for AMD/ATI VGA controller
                if 'VGA' in line and ('AMD' in line or 'ATI' in line):
                    # Check if it's a supported card
                    if any(model in line for model in ['Polaris', 'Ellesmere', '580', '570', '480', '470', 'Vega']):
                        # Extract PCI ID
                        pci_match = re.match(r'^([0-9a-f:\.]+)', line)
                        pci_id = pci_match.group(1) if pci_match else "unknown"
                        
                        # Extract device name
                        name_match = re.search(r'\[([^\]]+)\]', line)
                        device_name = name_match.group(1) if name_match else "AMD GPU"
                        
                        # Determine architecture
                        if 'Polaris' in line or 'Ellesmere' in line or any(x in line for x in ['580', '570', '480', '470']):
                            architecture = "GCN 4.0 (Polaris)"
                            family = "polaris"
                        elif 'Vega' in line:
                            architecture = "GCN 5.0 (Vega)"
                            family = "vega"
                        else:
                            architecture = "GCN (Unknown)"
                            family = "unknown"
                        
                        gpu_info = GPUInfo(
                            device_name=device_name,
                            pci_id=pci_id,
                            architecture=architecture,
                            family=family
                        )
                        
                        # Check for driver info
                        for j in range(i + 1, min(i + 15, len(lines))):
                            if 'Kernel driver in use:' in lines[j]:
                                gpu_info.driver_version = lines[j].split(':')[1].strip()
                                break
                        
                        return gpu_info
            
            return None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
    
    def _detect_via_rocm(self) -> Optional[GPUInfo]:
        """Detect GPU using ROCm SMI."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse output for GPU name
                for line in result.stdout.split('\n'):
                    if 'Card series' in line or 'GPU' in line:
                        name = line.split(':')[-1].strip()
                        
                        # Determine family
                        if any(x in name.lower() for x in ['580', '570', '480', '470', 'polaris']):
                            family = "polaris"
                            architecture = "GCN 4.0 (Polaris)"
                        elif 'vega' in name.lower():
                            family = "vega"
                            architecture = "GCN 5.0 (Vega)"
                        else:
                            family = "unknown"
                            architecture = "GCN"
                        
                        return GPUInfo(
                            device_name=name,
                            pci_id="unknown",
                            architecture=architecture,
                            family=family
                        )
            
            return None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
    
    def _detect_via_opencl(self) -> Optional[GPUInfo]:
        """Detect GPU using OpenCL."""
        try:
            result = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'AMD' in result.stdout:
                # Basic detection - OpenCL found AMD device
                return GPUInfo(
                    device_name="AMD GPU (via OpenCL)",
                    pci_id="unknown",
                    architecture="GCN",
                    family="unknown"
                )
            
            return None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
    
    def _classify_gpu_family(self, gpu_info: GPUInfo):
        """Classify GPU and enrich with family-specific data."""
        try:
            from src.core.gpu_family import detect_gpu_family
            
            self._gpu_family = detect_gpu_family()
            
            # Enrich GPUInfo with family data
            gpu_info.compute_units = self._gpu_family.compute_units
            gpu_info.wavefront_size = self._gpu_family.capabilities.wavefront_size
            gpu_info.memory_bandwidth_gbps = self._gpu_family.capabilities.memory_bandwidth_gbps
            gpu_info.fp32_tflops = self._gpu_family.capabilities.fp32_tflops
            gpu_info.recommended_batch_size = self._gpu_family.recommended_batch_size
            gpu_info.family = self._gpu_family.name
            
        except ImportError:
            # gpu_family module not available, use basic detection
            pass
    
    def _check_opencl(self) -> Tuple[bool, Optional[str]]:
        """
        Check if OpenCL is available and get version.
        
        Returns:
            Tuple of (available: bool, version: str or None)
        """
        try:
            result = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and "Platform" in result.stdout:
                # Try to extract OpenCL version
                version_match = re.search(r'OpenCL\s+([\d\.]+)', result.stdout)
                version = version_match.group(1) if version_match else None
                return True, version
            
            return False, None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False, None
    
    def _check_rocm(self) -> Tuple[bool, Optional[str]]:
        """
        Check if ROCm is available and get version.
        
        Returns:
            Tuple of (available: bool, version: str or None)
        """
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Try to get ROCm version
                try:
                    version_result = subprocess.run(
                        ["rocm-smi", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    version_match = re.search(r'(\d+\.\d+\.\d+)', version_result.stdout)
                    version = version_match.group(1) if version_match else None
                except:
                    version = None
                
                return True, version
            
            return False, None
            
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False, None
    
    def _get_vram_gb(self) -> float:
        """
        Get VRAM size in GB.
        
        Returns:
            VRAM in gigabytes
        """
        try:
            # Method 1: Try sysfs (most accurate)
            drm_paths = [
                '/sys/class/drm/card0/device/mem_info_vram_total',
                '/sys/class/drm/card1/device/mem_info_vram_total',
            ]
            
            for path in drm_paths:
                try:
                    with open(path, 'r') as f:
                        vram_bytes = int(f.read().strip())
                        return vram_bytes / (1024 ** 3)  # Convert to GB
                except (FileNotFoundError, ValueError, PermissionError):
                    continue
            
            # Method 2: Try rocm-smi
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Parse VRAM from output
                    match = re.search(r'(\d+)\s*MB', result.stdout)
                    if match:
                        return int(match.group(1)) / 1024
            except:
                pass
            
            # Fallback: Check GPU family
            if self._gpu_family:
                return self._gpu_family.vram_gb
            
            # Default for most common Polaris
            return 8.0
            
        except Exception:
            return 8.0  # Safe default
    
    def _calculate_performance_metrics(self, gpu_info: GPUInfo):
        """Calculate expected performance metrics based on hardware."""
        if gpu_info.compute_units > 0 and gpu_info.max_clock_mhz == 0:
            # Estimate clock for common Polaris cards
            if '580' in gpu_info.device_name or '480' in gpu_info.device_name:
                gpu_info.max_clock_mhz = 1340  # Typical boost clock
            elif '570' in gpu_info.device_name or '470' in gpu_info.device_name:
                gpu_info.max_clock_mhz = 1270
            else:
                gpu_info.max_clock_mhz = 1200  # Conservative estimate
        
        # Calculate TFLOPS if not set
        if gpu_info.fp32_tflops == 0 and gpu_info.compute_units > 0:
            # Formula: (CUs * 64 SIMD lanes * 2 ops/clock * clock_ghz) / 1000
            clock_ghz = gpu_info.max_clock_mhz / 1000.0
            gpu_info.fp32_tflops = (gpu_info.compute_units * 64 * 2 * clock_ghz) / 1000.0
    
    def initialize(self) -> bool:
        """
        Initialize GPU for computation with GCN-specific optimizations.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self._gpu_info:
            try:
                self._gpu_info = self.detect_gpu()
            except GPUDetectionError as e:
                print(f"❌ GPU initialization failed: {e}")
                return False
        
        print(f"\n{'='*70}")
        print(f"  Legacy GPU AI Platform - GPU Initialization")
        print(f"{'='*70}")
        print(f"  Device: {self._gpu_info.device_name}")
        print(f"  Architecture: {self._gpu_info.architecture}")
        print(f"  PCI ID: {self._gpu_info.pci_id}")
        print(f"  Driver: {self._gpu_info.driver_version or 'Unknown'}")
        print(f"{'─'*70}")
        print(f"  Hardware Specs:")
        print(f"    • VRAM: {self._gpu_info.memory_total_gb:.1f} GB")
        print(f"    • Compute Units: {self._gpu_info.compute_units}")
        print(f"    • Wavefront Size: {self._gpu_info.wavefront_size}")
        print(f"    • Memory Bandwidth: {self._gpu_info.memory_bandwidth_gbps:.0f} GB/s")
        print(f"    • FP32 Performance: {self._gpu_info.fp32_tflops:.2f} TFLOPS")
        print(f"{'─'*70}")
        print(f"  Compute Backends:")
        if self._gpu_info.opencl_available:
            print(f"    ✅ OpenCL {self._gpu_info.opencl_version or ''}")
        else:
            print(f"    ❌ OpenCL not available")
        
        if self._gpu_info.rocm_available:
            print(f"    ✅ ROCm {self._gpu_info.rocm_version or ''}")
        else:
            print(f"    ⚠️  ROCm not available (OpenCL will be used)")
        print(f"{'─'*70}")
        print(f"  Recommended Settings:")
        print(f"    • Batch Size: {self._gpu_info.recommended_batch_size}")
        print(f"    • Precision: FP32 (no FP16 acceleration on Polaris)")
        print(f"{'='*70}\n")
        
        # Initialize OpenCL context if available
        if self._gpu_info.opencl_available:
            try:
                self._init_opencl_context()
            except Exception as e:
                print(f"⚠️  OpenCL context initialization failed: {e}")
        
        self._initialized = True
        return True
    
    def _init_opencl_context(self):
        """Initialize OpenCL context with GCN optimizations."""
        # This would use pyopencl if available
        # For now, we just mark it as attempted
        try:
            import pyopencl as cl
            
            # Get AMD platform
            platforms = cl.get_platforms()
            amd_platform = None
            
            for platform in platforms:
                if 'AMD' in platform.name or 'Advanced Micro Devices' in platform.name:
                    amd_platform = platform
                    break
            
            if amd_platform:
                devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    # Create context with first GPU device
                    self._opencl_context = cl.Context(devices=[devices[0]])
                    print(f"    ✅ OpenCL context created for {devices[0].name}")
        
        except ImportError:
            # pyopencl not installed, that's okay
            pass
        except Exception as e:
            # Log but don't fail
            print(f"    ⚠️  Could not create OpenCL context: {e}")
    
    def get_info(self) -> Dict[str, any]:
        """
        Get comprehensive GPU information as dictionary.
        
        Returns:
            Dictionary with all GPU information
        """
        if not self._gpu_info:
            return {"error": "GPU not detected"}
        
        return {
            "device_name": self._gpu_info.device_name,
            "pci_id": self._gpu_info.pci_id,
            "architecture": self._gpu_info.architecture,
            "family": self._gpu_info.family,
            "driver_version": self._gpu_info.driver_version,
            
            # Memory
            "memory_total_gb": self._gpu_info.memory_total_gb,
            "memory_bandwidth_gbps": self._gpu_info.memory_bandwidth_gbps,
            
            # Compute
            "compute_units": self._gpu_info.compute_units,
            "wavefront_size": self._gpu_info.wavefront_size,
            "max_clock_mhz": self._gpu_info.max_clock_mhz,
            "fp32_tflops": self._gpu_info.fp32_tflops,
            
            # Backends
            "opencl_available": self._gpu_info.opencl_available,
            "opencl_version": self._gpu_info.opencl_version,
            "rocm_available": self._gpu_info.rocm_available,
            "rocm_version": self._gpu_info.rocm_version,
            
            # Recommendations
            "recommended_batch_size": self._gpu_info.recommended_batch_size,
        }
    
    def get_gpu_info(self) -> Optional[GPUInfo]:
        """Get raw GPUInfo object."""
        return self._gpu_info
    
    def is_initialized(self) -> bool:
        """Check if GPU is initialized."""
        return self._initialized
    
    def get_compute_backend(self) -> str:
        """
        Determine best available compute backend.
        
        Returns:
            'rocm', 'opencl', or 'cpu'
        """
        if not self._gpu_info:
            return 'cpu'
        
        if self._gpu_info.rocm_available:
            return 'rocm'
        elif self._gpu_info.opencl_available:
            return 'opencl'
        else:
            return 'cpu'
    
    def get_opencl_context(self):
        """Get OpenCL context if available."""
        return self._opencl_context
    
    def get_optimization_hints(self) -> Dict[str, any]:
        """
        Get GCN-specific optimization hints.
        
        Returns:
            Dictionary with optimization recommendations
        """
        if not self._gpu_info:
            return {}
        
        hints = {
            "wavefront_size": self._gpu_info.wavefront_size,
            "preferred_workgroup_multiple": self._gpu_info.wavefront_size,
            "max_workgroup_size": 256,  # GCN typical
            "local_memory_size_kb": 64,  # GCN LDS
            
            # Memory access patterns
            "coalesced_access_bytes": 128,  # GCN memory controller
            "cache_line_bytes": 64,
            
            # Execution hints
            "async_copy_preferred": True,
            "vectorization_width": 4,  # float4 operations
            
            # Polaris-specific
            "sparse_operations_beneficial": self._gpu_info.memory_total_gb <= 8.0,
            "fp16_acceleration": False,  # No on Polaris
            "int8_emulated": True,
        }
        
        return hints
