#!/usr/bin/env python3
"""
Task 1.1.1: Diagnose FLOAT4 Kernel Compilation Error
Phase 1: Quick Wins - Roadmap de Optimización

Este script diagnostica por qué el kernel FLOAT4 falla en Clover OpenCL 1.1
"""

import pyopencl as cl
import numpy as np
import sys
from pathlib import Path

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_opencl_version(device):
    """Check OpenCL version support"""
    print_section("OpenCL Version Check")
    
    version = device.version
    opencl_version = device.opencl_c_version
    
    print(f"Device: {device.name}")
    print(f"Driver Version: {version}")
    print(f"OpenCL C Version: {opencl_version}")
    
    # Check specific features
    extensions = device.extensions.split()
    print(f"\nRelevant Extensions:")
    for ext in extensions:
        if any(keyword in ext.lower() for keyword in ['float', 'double', 'vector']):
            print(f"  • {ext}")
    
    return version, opencl_version

def test_simple_float4():
    """Test a simple float4 kernel"""
    print_section("Test 1: Simple Float4 Kernel")
    
    simple_kernel = """
    __kernel void test_float4(__global const float4* A, __global float4* B) {
        int gid = get_global_id(0);
        B[gid] = A[gid] * 2.0f;
    }
    """
    
    try:
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices(cl.device_type.GPU)
        ctx = cl.Context([devices[0]])
        
        prg = cl.Program(ctx, simple_kernel).build(options='-cl-std=CL1.1')
        print("✅ Simple float4 kernel compiles successfully")
        return True, ctx, devices[0]
    except Exception as e:
        print(f"❌ Simple float4 kernel FAILED:")
        print(f"   Error: {str(e)[:300]}")
        return False, None, None

def test_restrict_keyword(ctx):
    """Test restrict keyword support"""
    print_section("Test 2: Restrict Keyword (OpenCL 1.2+)")
    
    restrict_kernel = """
    __kernel void test_restrict(
        __global const float* restrict A,
        __global float* restrict B
    ) {
        int gid = get_global_id(0);
        B[gid] = A[gid] * 2.0f;
    }
    """
    
    try:
        prg = cl.Program(ctx, restrict_kernel).build(options='-cl-std=CL1.1')
        print("✅ 'restrict' keyword works in CL1.1")
        return True
    except Exception as e:
        print(f"❌ 'restrict' keyword NOT supported in CL1.1:")
        print(f"   Error: {str(e)[:200]}")
        return False

def test_float4_with_local_memory(ctx):
    """Test float4 with local memory"""
    print_section("Test 3: Float4 with Local Memory")
    
    kernel = """
    __kernel void test_float4_local(
        __global const float4* A,
        __global float4* B,
        __local float4* tile
    ) {
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        
        tile[lid] = A[gid];
        barrier(CLK_LOCAL_MEM_FENCE);
        B[gid] = tile[lid] * 2.0f;
    }
    """
    
    try:
        prg = cl.Program(ctx, kernel).build(options='-cl-std=CL1.1')
        print("✅ Float4 with local memory works")
        return True
    except Exception as e:
        print(f"❌ Float4 with local memory FAILED:")
        print(f"   Error: {str(e)[:200]}")
        return False

def test_actual_gemm_float4(ctx):
    """Test the actual GEMM FLOAT4 kernel from the codebase"""
    print_section("Test 4: Actual GEMM FLOAT4 Kernel")
    
    kernel_file = Path(__file__).parent.parent / 'src/opencl/kernels/gemm_rx580_optimized.cl'
    
    if not kernel_file.exists():
        print(f"❌ Kernel file not found: {kernel_file}")
        return False
    
    with open(kernel_file, 'r') as f:
        kernel_source = f.read()
    
    print(f"Loading kernel from: {kernel_file}")
    print(f"Kernel source size: {len(kernel_source)} bytes")
    
    # Try to compile with CL1.1
    try:
        prg = cl.Program(ctx, kernel_source).build(options='-cl-std=CL1.1')
        print("✅ GEMM FLOAT4 kernel compiles successfully with CL1.1!")
        return True
    except cl.RuntimeError as e:
        error_msg = str(e)
        print(f"❌ GEMM FLOAT4 kernel compilation FAILED:")
        print(f"\nFull error message:")
        print("─" * 70)
        print(error_msg)
        print("─" * 70)
        
        # Analyze specific errors
        print("\n🔍 Error Analysis:")
        
        if 'restrict' in error_msg.lower():
            print("  ⚠️  Issue: 'restrict' keyword not supported in OpenCL 1.1")
            print("  💡 Solution: Remove '__restrict' qualifiers")
        
        if 'pragma' in error_msg.lower():
            print("  ⚠️  Issue: Pragma directive issue")
            print("  💡 Solution: Check pragma syntax for CL1.1 compatibility")
        
        if 'local' in error_msg.lower() and 'memory' in error_msg.lower():
            print("  ⚠️  Issue: Local memory allocation problem")
            print("  💡 Solution: Verify LDS size limits and allocation")
        
        if 'vector' in error_msg.lower() or 'float4' in error_msg.lower():
            print("  ⚠️  Issue: Vector type issue")
            print("  💡 Solution: Check float4 usage patterns")
        
        if 'undefined' in error_msg.lower():
            print("  ⚠️  Issue: Undefined symbol or constant")
            print("  💡 Solution: Check all #define macros are set")
        
        return False

def test_alternative_vectorization(ctx):
    """Test alternative vectorization without float4 pointers"""
    print_section("Test 5: Alternative Vectorization (vload4/vstore4)")
    
    alt_kernel = """
    __kernel void test_vload_vstore(
        __global const float* A,
        __global float* B,
        int N
    ) {
        int gid = get_global_id(0);
        int idx = gid * 4;
        
        if (idx + 3 < N) {
            float4 vec = vload4(gid, A);
            vec *= 2.0f;
            vstore4(vec, gid, B);
        }
    }
    """
    
    try:
        prg = cl.Program(ctx, alt_kernel).build(options='-cl-std=CL1.1')
        print("✅ vload4/vstore4 approach works!")
        print("   💡 This could be a good alternative for Clover")
        return True
    except Exception as e:
        print(f"❌ vload4/vstore4 approach FAILED:")
        print(f"   Error: {str(e)[:200]}")
        return False

def generate_recommendations(results):
    """Generate recommendations based on test results"""
    print_section("Recommendations for Clover Compatibility")
    
    print("Based on the diagnostic tests, here are the recommended fixes:\n")
    
    if not results.get('restrict', True):
        print("1. 🔧 Remove '__restrict' qualifiers")
        print("   - OpenCL 1.1 doesn't support the 'restrict' keyword")
        print("   - Replace: __global const float* restrict A")
        print("   - With:    __global const float* A")
        print()
    
    if results.get('vload_vstore', False):
        print("2. 🔧 Use vload4/vstore4 instead of float4 pointers")
        print("   - Clover may have better support for vload/vstore")
        print("   - Replace: __global const float4* A")
        print("   - With:    __global const float* A + vload4()")
        print()
    
    print("3. 🔧 Simplify pragma directives")
    print("   - Use only basic pragmas compatible with CL1.1")
    print("   - Test: #pragma unroll 4")
    print()
    
    print("4. 🔧 Verify LDS allocation")
    print("   - Ensure __local memory is passed as kernel argument")
    print("   - Don't rely on automatic allocation")
    print()
    
    print("5. 🔧 Check constant definitions")
    print("   - All #define macros should be defined before use")
    print("   - Consider passing some as kernel arguments")
    print()

def main():
    """Main diagnostic function"""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " FLOAT4 Kernel Diagnostic Tool".center(68) + "║")
    print("║" + " Task 1.1.1 - Phase 1: Quick Wins".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    try:
        # Test 1: Check OpenCL version
        platforms = cl.get_platforms()
        if not platforms:
            print("❌ No OpenCL platforms found!")
            return 1
        
        devices = platforms[0].get_devices(cl.device_type.GPU)
        if not devices:
            print("❌ No GPU devices found!")
            return 1
        
        check_opencl_version(devices[0])
        
        # Test 2: Simple float4
        success, ctx, device = test_simple_float4()
        results['simple_float4'] = success
        
        if not ctx:
            print("\n❌ Cannot continue without OpenCL context")
            return 1
        
        # Test 3: Restrict keyword
        results['restrict'] = test_restrict_keyword(ctx)
        
        # Test 4: Float4 with local memory
        results['local_memory'] = test_float4_with_local_memory(ctx)
        
        # Test 5: Actual GEMM kernel
        results['gemm_float4'] = test_actual_gemm_float4(ctx)
        
        # Test 6: Alternative vectorization
        results['vload_vstore'] = test_alternative_vectorization(ctx)
        
        # Generate recommendations
        generate_recommendations(results)
        
        # Summary
        print_section("Test Summary")
        print(f"Simple float4:          {'✅ PASS' if results.get('simple_float4') else '❌ FAIL'}")
        print(f"Restrict keyword:       {'✅ PASS' if results.get('restrict') else '❌ FAIL'}")
        print(f"Local memory:           {'✅ PASS' if results.get('local_memory') else '❌ FAIL'}")
        print(f"GEMM FLOAT4 kernel:     {'✅ PASS' if results.get('gemm_float4') else '❌ FAIL'}")
        print(f"vload4/vstore4:         {'✅ PASS' if results.get('vload_vstore') else '❌ FAIL'}")
        
        if results.get('gemm_float4'):
            print("\n🎉 Success! GEMM FLOAT4 kernel is compatible with Clover!")
            return 0
        else:
            print("\n⚠️  GEMM FLOAT4 kernel needs fixes for Clover compatibility")
            print("    Proceed to Task 1.1.2 to implement the fixes")
            return 0  # Not a failure, just identified issues
        
    except Exception as e:
        print(f"\n❌ Unexpected error during diagnostics:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
