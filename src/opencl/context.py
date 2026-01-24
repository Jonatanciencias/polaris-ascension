"""
OpenCL Context Management
=========================

Manages OpenCL device discovery, context creation, and command queue lifecycle.

This module provides a clean abstraction over PyOpenCL for managing GPU resources.
It handles device selection, memory management, and ensures proper cleanup.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    logging.warning("PyOpenCL not available. Install with: pip install pyopencl")


logger = logging.getLogger(__name__)


@dataclass
class CLDevice:
    """
    Represents an OpenCL device with its capabilities.
    
    Attributes:
        name: Device name (e.g., "Radeon RX 580")
        vendor: Vendor name (e.g., "Advanced Micro Devices, Inc.")
        version: OpenCL version supported
        compute_units: Number of compute units
        max_work_group_size: Maximum work-group size
        local_mem_size: Local memory size in bytes
        global_mem_size: Global memory size in bytes
        device: PyOpenCL device object
    """
    name: str
    vendor: str
    version: str
    compute_units: int
    max_work_group_size: int
    local_mem_size: int
    global_mem_size: int
    device: 'cl.Device'
    
    def __str__(self):
        return (
            f"CLDevice(\n"
            f"  name='{self.name}'\n"
            f"  vendor='{self.vendor}'\n"
            f"  version='{self.version}'\n"
            f"  compute_units={self.compute_units}\n"
            f"  max_work_group_size={self.max_work_group_size}\n"
            f"  local_mem={self.local_mem_size / 1024:.0f} KB\n"
            f"  global_mem={self.global_mem_size / (1024**3):.2f} GB\n"
            f")"
        )


class CLContext:
    """
    OpenCL context manager for GPU operations.
    
    This class manages the lifecycle of OpenCL resources including:
    - Device selection and context creation
    - Command queue management
    - Kernel compilation and caching
    - Memory buffer allocation
    
    Example:
        # Automatic device selection
        ctx = CLContext()
        
        # Manual device selection
        devices = CLContext.list_devices()
        ctx = CLContext(device_id=0)
        
        # Use context
        with ctx:
            result = gemm(ctx, A, B)
    """
    
    def __init__(self, device_id: Optional[int] = None, enable_profiling: bool = False):
        """
        Initialize OpenCL context.
        
        Args:
            device_id: Specific device ID to use. If None, selects first GPU.
            enable_profiling: Enable command queue profiling for performance analysis.
            
        Raises:
            RuntimeError: If OpenCL is not available or no devices found.
        """
        if not OPENCL_AVAILABLE:
            raise RuntimeError(
                "PyOpenCL not available. Install with: pip install pyopencl"
            )
        
        self.enable_profiling = enable_profiling
        self._context: Optional[cl.Context] = None
        self._queue: Optional[cl.CommandQueue] = None
        self._device_info: Optional[CLDevice] = None
        self._kernel_cache: dict = {}
        
        # Initialize context
        self._initialize_context(device_id)
        
        logger.info(f"Initialized OpenCL context on device: {self.device.name}")
    
    def _initialize_context(self, device_id: Optional[int]):
        """Initialize OpenCL context and command queue."""
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        # Find all GPU devices
        gpu_devices = []
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                gpu_devices.extend(devices)
            except cl.RuntimeError:
                continue
        
        if not gpu_devices:
            # Fallback to CPU if no GPU available
            logger.warning("No GPU devices found, falling back to CPU")
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type=cl.device_type.CPU)
                    gpu_devices.extend(devices)
                    break
                except cl.RuntimeError:
                    continue
        
        if not gpu_devices:
            raise RuntimeError("No OpenCL devices found")
        
        # Select device
        if device_id is not None:
            if device_id >= len(gpu_devices):
                raise ValueError(
                    f"Device ID {device_id} out of range. "
                    f"Available devices: {len(gpu_devices)}"
                )
            device = gpu_devices[device_id]
        else:
            # Select first AMD GPU if available, otherwise first device
            device = None
            for dev in gpu_devices:
                vendor = dev.vendor.lower()
                if 'amd' in vendor or 'advanced micro devices' in vendor:
                    device = dev
                    break
            if device is None:
                device = gpu_devices[0]
        
        # Create context and queue
        self._context = cl.Context([device])
        
        queue_properties = cl.command_queue_properties.PROFILING_ENABLE if self.enable_profiling else None
        self._queue = cl.CommandQueue(self._context, properties=queue_properties)
        
        # Store device info
        self._device_info = CLDevice(
            name=device.name.strip(),
            vendor=device.vendor.strip(),
            version=device.version.strip(),
            compute_units=device.max_compute_units,
            max_work_group_size=device.max_work_group_size,
            local_mem_size=device.local_mem_size,
            global_mem_size=device.global_mem_size,
            device=device
        )
    
    @staticmethod
    def list_devices() -> List[CLDevice]:
        """
        List all available OpenCL devices.
        
        Returns:
            List of CLDevice objects representing available devices.
        """
        if not OPENCL_AVAILABLE:
            return []
        
        devices = []
        platforms = cl.get_platforms()
        
        for platform in platforms:
            try:
                platform_devices = platform.get_devices()
                for device in platform_devices:
                    devices.append(CLDevice(
                        name=device.name.strip(),
                        vendor=device.vendor.strip(),
                        version=device.version.strip(),
                        compute_units=device.max_compute_units,
                        max_work_group_size=device.max_work_group_size,
                        local_mem_size=device.local_mem_size,
                        global_mem_size=device.global_mem_size,
                        device=device
                    ))
            except cl.RuntimeError:
                continue
        
        return devices
    
    @property
    def context(self) -> 'cl.Context':
        """Get OpenCL context."""
        return self._context
    
    @property
    def queue(self) -> 'cl.CommandQueue':
        """Get command queue."""
        return self._queue
    
    @property
    def device(self) -> CLDevice:
        """Get device information."""
        return self._device_info
    
    def create_buffer(self, flags: 'cl.mem_flags', size: int) -> 'cl.Buffer':
        """
        Create an OpenCL buffer.
        
        Args:
            flags: Memory flags (e.g., cl.mem_flags.READ_WRITE)
            size: Buffer size in bytes
            
        Returns:
            OpenCL buffer object
        """
        return cl.Buffer(self._context, flags, size)
    
    def create_buffer_from_array(
        self, 
        array: np.ndarray, 
        flags: Optional['cl.mem_flags'] = None
    ) -> 'cl.Buffer':
        """
        Create OpenCL buffer from numpy array.
        
        Args:
            array: Numpy array to copy to device
            flags: Memory flags. Defaults to READ_ONLY.
            
        Returns:
            OpenCL buffer containing array data
        """
        if flags is None:
            flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        
        return cl.Buffer(
            self._context,
            flags,
            hostbuf=np.ascontiguousarray(array)
        )
    
    def compile_kernel(self, source: str, kernel_name: str) -> 'cl.Kernel':
        """
        Compile OpenCL kernel from source.
        
        Args:
            source: OpenCL kernel source code
            kernel_name: Name of kernel function
            
        Returns:
            Compiled kernel object
            
        Note:
            Kernels are cached to avoid recompilation.
        """
        cache_key = (kernel_name, hash(source))
        
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]
        
        try:
            program = cl.Program(self._context, source).build()
            kernel = getattr(program, kernel_name)
            self._kernel_cache[cache_key] = kernel
            logger.debug(f"Compiled kernel: {kernel_name}")
            return kernel
        except cl.RuntimeError as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise
    
    def load_kernel_from_file(self, filepath: str, kernel_name: str) -> 'cl.Kernel':
        """
        Load and compile kernel from .cl file.
        
        Args:
            filepath: Path to .cl kernel file
            kernel_name: Name of kernel function
            
        Returns:
            Compiled kernel object
        """
        with open(filepath, 'r') as f:
            source = f.read()
        return self.compile_kernel(source, kernel_name)
    
    def finish(self):
        """Wait for all queued operations to complete."""
        self._queue.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure queue finishes."""
        if self._queue is not None:
            self._queue.finish()
        return False
    
    def __del__(self):
        """Cleanup on destruction."""
        if self._queue is not None:
            try:
                self._queue.finish()
            except:
                pass
    
    def __repr__(self):
        return f"CLContext(device='{self.device.name}')"
