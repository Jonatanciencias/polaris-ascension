#!/usr/bin/env python3
"""
Direct OpenCL test bypassing PyOpenCL's automatic compilation
to diagnose Mesa Clover issues.

This script uses ctypes to call libOpenCL.so directly.
"""

import ctypes
import ctypes.util
import sys
from pathlib import Path

# Find libOpenCL.so
opencl_lib_path = ctypes.util.find_library("OpenCL")
if not opencl_lib_path:
    print("❌ libOpenCL.so not found")
    sys.exit(1)

print(f"📚 Found OpenCL: {opencl_lib_path}")
cl = ctypes.CDLL(opencl_lib_path)

# OpenCL constants
CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 1 << 2
CL_DEVICE_NAME = 0x102B
CL_DEVICE_COMPUTE_UNITS = 0x1002
CL_PROGRAM_BUILD_LOG = 0x1183

# Define ctypes structures
cl_platform_id = ctypes.c_void_p
cl_device_id = ctypes.c_void_p
cl_context = ctypes.c_void_p
cl_program = ctypes.c_void_p
cl_int = ctypes.c_int32
cl_uint = ctypes.c_uint32

# Function signatures
cl.clGetPlatformIDs.argtypes = [cl_uint, ctypes.POINTER(cl_platform_id), ctypes.POINTER(cl_uint)]
cl.clGetPlatformIDs.restype = cl_int

cl.clGetDeviceIDs.argtypes = [cl_platform_id, ctypes.c_uint64, cl_uint, ctypes.POINTER(cl_device_id), ctypes.POINTER(cl_uint)]
cl.clGetDeviceIDs.restype = cl_int

cl.clGetDeviceInfo.argtypes = [cl_device_id, cl_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
cl.clGetDeviceInfo.restype = cl_int

cl.clCreateContext.argtypes = [ctypes.c_void_p, cl_uint, ctypes.POINTER(cl_device_id), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(cl_int)]
cl.clCreateContext.restype = cl_context

cl.clCreateProgramWithSource.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(cl_int)]
cl.clCreateProgramWithSource.restype = cl_program

cl.clBuildProgram.argtypes = [cl_program, cl_uint, ctypes.POINTER(cl_device_id), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
cl.clBuildProgram.restype = cl_int

cl.clGetProgramBuildInfo.argtypes = [cl_program, cl_device_id, cl_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
cl.clGetProgramBuildInfo.restype = cl_int

def main():
    print("\n🔬 Test directo de OpenCL C API\n")
    
    # 1. Get platforms
    num_platforms = cl_uint()
    cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
    print(f"Plataformas disponibles: {num_platforms.value}")
    
    platforms = (cl_platform_id * num_platforms.value)()
    cl.clGetPlatformIDs(num_platforms.value, platforms, None)
    
    # 2. Get GPU device
    device = None
    for platform in platforms:
        num_devices = cl_uint()
        ret = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(num_devices))
        if ret == CL_SUCCESS and num_devices.value > 0:
            devices = (cl_device_id * num_devices.value)()
            cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices.value, devices, None)
            device = devices[0]
            break
    
    if not device:
        print("❌ No se encontró GPU")
        return
    
    # Get device name
    name_size = ctypes.c_size_t()
    cl.clGetDeviceInfo(device, CL_DEVICE_NAME, 0, None, ctypes.byref(name_size))
    name_buf = ctypes.create_string_buffer(name_size.value)
    cl.clGetDeviceInfo(device, CL_DEVICE_NAME, name_size.value, name_buf, None)
    print(f"✅ GPU encontrada: {name_buf.value.decode()}")
    
    # 3. Create context
    err = cl_int()
    context = cl.clCreateContext(None, 1, (cl_device_id * 1)(device), None, None, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        print(f"❌ Error creando contexto: {err.value}")
        return
    print("✅ Contexto creado")
    
    # 4. Load and compile kernel
    kernel_path = Path(__file__).parent.parent / "src" / "opencl" / "kernels" / "gemm.cl"
    with open(kernel_path, 'r') as f:
        source = f.read()
    
    print(f"\n📝 Compilando kernel ({len(source)} bytes)...")
    
    source_c = ctypes.c_char_p(source.encode('utf-8'))
    source_len = ctypes.c_size_t(len(source))
    
    program = cl.clCreateProgramWithSource(
        context,
        1,
        ctypes.byref(source_c),
        ctypes.byref(source_len),
        ctypes.byref(err)
    )
    
    if err.value != CL_SUCCESS:
        print(f"❌ Error creando programa: {err.value}")
        return
    
    # Build with minimal options
    build_options = b"-cl-std=CL1.2"
    ret = cl.clBuildProgram(program, 1, (cl_device_id * 1)(device), build_options, None, None)
    
    # Get build log
    log_size = ctypes.c_size_t()
    cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size))
    
    if log_size.value > 1:
        log_buf = ctypes.create_string_buffer(log_size.value)
        cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log_buf, None)
        build_log = log_buf.value.decode('utf-8', errors='replace')
        print(f"\n📋 Build Log:")
        print(build_log)
    
    if ret == CL_SUCCESS:
        print("\n✅ ¡KERNEL COMPILADO EXITOSAMENTE!")
        print("🎉 Mesa Clover funciona correctamente")
        return True
    else:
        print(f"\n❌ Error de compilación: código {ret}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
