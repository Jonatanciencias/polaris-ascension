"""
GPU Management Module

Handles GPU detection, initialization, and resource management for AMD Radeon RX 580.
"""

import subprocess
import re
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU Information container"""
    name: str
    pci_id: str
    architecture: str
    vram_mb: Optional[int] = None
    driver: Optional[str] = None
    opencl_available: bool = False
    rocm_available: bool = False


class GPUManager:
    """Manages AMD Radeon RX 580 GPU resources"""
    
    def __init__(self):
        self._gpu_info: Optional[GPUInfo] = None
        self._initialized = False
        
    def detect_gpu(self) -> Optional[GPUInfo]:
        """
        Detect AMD Radeon RX 580 GPU in the system.
        
        Returns:
            GPUInfo object if GPU is detected, None otherwise
        """
        try:
            # Run lspci to detect GPU
            result = subprocess.run(
                ["lspci", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.split('\n')
            gpu_info = None
            
            for i, line in enumerate(lines):
                # Look for Radeon RX 580 or Polaris 20
                if 'VGA' in line and ('Radeon RX 580' in line or 'Polaris 20' in line):
                    # Extract PCI ID
                    pci_match = re.match(r'^([0-9a-f:\.]+)', line)
                    pci_id = pci_match.group(1) if pci_match else "unknown"
                    
                    # Extract GPU name
                    name_match = re.search(r'\[([^\]]+)\]', line)
                    name = name_match.group(1) if name_match else "AMD Radeon RX 580"
                    
                    gpu_info = GPUInfo(
                        name=name,
                        pci_id=pci_id,
                        architecture="Polaris 20 (GCN 4.0)"
                    )
                    
                    # Check for driver info in subsequent lines
                    for j in range(i + 1, min(i + 10, len(lines))):
                        if 'Kernel driver in use:' in lines[j]:
                            driver = lines[j].split(':')[1].strip()
                            gpu_info.driver = driver
                            break
                    
                    break
            
            if gpu_info:
                # Check OpenCL availability
                gpu_info.opencl_available = self._check_opencl()
                
                # Check ROCm availability
                gpu_info.rocm_available = self._check_rocm()
                
                # Try to get VRAM info
                gpu_info.vram_mb = self._get_vram()
                
                self._gpu_info = gpu_info
                
            return gpu_info
            
        except subprocess.CalledProcessError as e:
            print(f"Error detecting GPU: {e}")
            return None
    
    def _check_opencl(self) -> bool:
        """Check if OpenCL is available"""
        try:
            result = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and "Platform" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_rocm(self) -> bool:
        """Check if ROCm is available"""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_vram(self) -> Optional[int]:
        """Try to get VRAM size in MB"""
        try:
            # Try to read from sysfs
            with open('/sys/class/drm/card0/device/mem_info_vram_total', 'r') as f:
                vram_bytes = int(f.read().strip())
                return vram_bytes // (1024 * 1024)
        except (FileNotFoundError, ValueError, PermissionError):
            pass
        
        # Default for RX 580 variants
        return 8192  # Most common: 8GB
    
    def initialize(self) -> bool:
        """
        Initialize GPU for computation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self._gpu_info:
            self._gpu_info = self.detect_gpu()
        
        if not self._gpu_info:
            print("No compatible GPU detected")
            return False
        
        print(f"Initializing GPU: {self._gpu_info.name}")
        print(f"  PCI ID: {self._gpu_info.pci_id}")
        print(f"  Architecture: {self._gpu_info.architecture}")
        print(f"  Driver: {self._gpu_info.driver}")
        print(f"  VRAM: {self._gpu_info.vram_mb} MB")
        print(f"  OpenCL: {'Available' if self._gpu_info.opencl_available else 'Not available'}")
        print(f"  ROCm: {'Available' if self._gpu_info.rocm_available else 'Not available'}")
        
        self._initialized = True
        return True
    
    def get_info(self) -> Optional[GPUInfo]:
        """Get GPU information"""
        return self._gpu_info
    
    def is_initialized(self) -> bool:
        """Check if GPU is initialized"""
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
