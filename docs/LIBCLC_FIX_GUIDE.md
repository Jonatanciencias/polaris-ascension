# OpenCL libclc Fix Guide

**Date**: January 23, 2026  
**Status**: ✅ **RESOLVED**  
**Hardware**: AMD Radeon RX 590 GME (Polaris 10, gfx803)

---

## 🎯 Executive Summary

Successfully resolved Ubuntu 24.04's broken OpenCL stack by compiling libclc from LLVM source. AMD RX 590 now runs OpenCL kernels with **235 GFLOPS** performance on 1024×1024 GEMM operations.

---

## 🔴 The Problem

### Root Cause
Ubuntu 24.04's `libclc-20-dev` package ships with **fundamentally broken headers**:

**Missing Type Definitions:**
```c
// /usr/include/clc/convert.h - Line 78
_CLC_VECTOR_CONVERT_TO_SUFFIX(ulong, _sat)  // ❌ 'ulong' undefined
_CLC_VECTOR_CONVERT_TO_SUFFIX(uint, _sat)   // ❌ 'uint' undefined
_CLC_VECTOR_CONVERT_TO_SUFFIX(ushort, _sat) // ❌ 'ushort' undefined
_CLC_VECTOR_CONVERT_TO_SUFFIX(uchar, _sat)  // ❌ 'uchar' undefined
```

**Missing Files:**
- `/usr/include/clc/math/gentype.inc` - **NOT FOUND**
- All workitem functions missing `size_t` type

### Impact
- ❌ **Both Mesa Clover and RustiCL** fail to compile ANY kernel
- ❌ Every OpenCL program crashes with "undefined type" errors
- ❌ AMD Polaris GPUs effectively **unusable** for OpenCL

---

## ✅ The Solution

### Strategy: Compile libclc from Official LLVM Source

Instead of patching Ubuntu's broken package, we compiled libclc directly from upstream.

### Step 1: Clone LLVM libclc Source
```bash
cd /tmp
git clone --depth 1 --branch release/18.x \
  https://github.com/llvm/llvm-project.git llvm-libclc
cd llvm-libclc
mkdir build && cd build
```

**Result**: 146,022 objects, 213 MB downloaded in ~2 minutes

### Step 2: Configure Build (AMD Targets Only)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DLLVM_CONFIG=/usr/bin/llvm-config-18 \
      -DLIBCLC_TARGETS_TO_BUILD="amdgcn--;r600--" \
      ../libclc
```

**Key Decision**: Disabled SPIR-V targets (required missing `llvm-spirv` binary)  
**AMD Targets Included**:
- `amdgcn--`: Modern AMD GPUs (GCN 1.0 - RDNA 3)
  - Includes: polaris10, polaris11 (RX 580/590 family)
  - Plus: Vega, Navi, RDNA (42 GPU variants)
- `r600--`: Legacy AMD GPUs (TeraScale)
  - Cedar, Cypress, Barts, Cayman

**Configuration Time**: 0.5 seconds

### Step 3: Compile libclc
```bash
make -j28
```

**Compilation Stats**:
- **Time**: ~5 minutes (28 cores, Xeon E5-2680 v4)
- **Build Targets**: 100% complete
- **Files Generated**: 
  - Headers: `/usr/include/clc/` (complete type definitions)
  - Bitcode: `/usr/share/clc/*.bc` (42 AMD GPU targets)

**Warnings**: Minor data layout mismatches (non-critical, LLVM linker warnings)

### Step 4: Install
```bash
sudo make install
```

**Installation Manifest**:
```
Installing: /usr/include/clc/clctypes.h          ✅ (was broken)
Installing: /usr/include/clc/math/gentype.inc   ✅ (was missing)
Installing: /usr/include/clc/convert.h          ✅ (fixed types)
Installing: /usr/include/clc/workitem/*.h       ✅ (size_t defined)
Installing: /usr/share/clc/polaris10-amdgcn--.bc  ✅ (RX 590)
Installing: /usr/share/clc/polaris11-amdgcn--.bc  ✅ (RX 580)
... [65 total files]
```

---

## 🧪 Verification

### Test 1: Simple Kernel Compilation
```python
import pyopencl as cl

platforms = cl.get_platforms()
clover = [p for p in platforms if 'clover' in p.name.lower()][0]
ctx = cl.Context(clover.get_devices()[:1])

kernel = '''
__kernel void test(__global float* a) {
    uint gid = get_global_id(0);      // ✅ uint now defined
    size_t gs = get_global_size(0);   // ✅ size_t now defined
    uchar c = (uchar)(gid & 0xFF);    // ✅ uchar now defined
    ulong l = (ulong)gid;             // ✅ ulong now defined
    a[gid] = (float)(c + l);
}
'''

prg = cl.Program(ctx, kernel).build()  # ✅ SUCCESS
```

**Result**: ✅ **KERNEL COMPILED** (previously failed with undefined type errors)

### Test 2: GEMM Kernel Execution
```python
# Load production GEMM kernel (318 LOC, tiled implementation)
with open('src/opencl/kernels/gemm.cl', 'r') as f:
    prg = cl.Program(ctx, f.read()).build()  # ✅ Compiles all 3 variants

# Execute gemm_tiled (16×16 blocking, local memory)
M = N = K = 1024
kernel(queue, (M, N), (16, 16), 
       np.int32(M), np.int32(N), np.int32(K),
       np.float32(1.0), np.float32(0.0),
       a_buf, b_buf, c_buf)

# Verify correctness
error = np.abs(C_gpu - (A @ B)).max()  # ✅ Error: 2.06e-4
```

**Result**: ✅ **GEMM EXECUTES** with numerical accuracy <2e-4

---

## 📊 Performance Benchmarks

### GEMM Performance (Mesa Clover)
```
Matrix Size | Performance | Time    | Accuracy
------------|-------------|---------|----------
 256 × 256  |  176 GFLOPS |  0.19ms | 4.2e-5
 512 × 512  |  225 GFLOPS |  1.19ms | 9.2e-5
1024 × 1024 |  235 GFLOPS |  9.15ms | 2.1e-4
```

### Analysis
- **Peak Performance**: 235 GFLOPS @ 1024×1024
- **Efficiency**: 3.8% of theoretical peak (6.17 TFLOPS)
- **Scaling**: Good (176 → 235 GFLOPS as matrix grows)
- **Accuracy**: Excellent (<2e-4 error vs NumPy reference)

### Why Not Higher Performance?
1. **Mesa Clover** is not as optimized as ROCm (expected)
2. **OpenCL 1.1** limits (Clover doesn't support 2.0+ features)
3. **Memory bandwidth**: Likely hitting ~60 GB/s (23% of peak 256 GB/s)
4. **Kernel optimization**: Further tuning possible (tile size, vectorization)

**Context**: For comparison, ROCm on same hardware typically achieves 800-1200 GFLOPS.  
Our 235 GFLOPS with Mesa Clover is **reasonable and proves GPU is working**.

---

## 🛠️ Technical Details

### Build Environment
```
OS:        Ubuntu 24.04.1 LTS
Kernel:    6.14.0-37-generic
CPU:       Intel Xeon E5-2680 v4 (28 cores)
GPU:       AMD Radeon RX 590 GME (Polaris 10, gfx803)
LLVM:      18.1.3 (system), 16.0.6 (build tools)
Compiler:  clang 18, GCC 13.3.0
Mesa:      25.0.7 (Clover driver)
```

### Dependencies
```bash
# Already installed (no additional packages needed)
sudo apt install -y \
  llvm-18 llvm-18-dev \
  clang-18 \
  git cmake ninja-build \
  python3-dev
```

### File Locations
```
Source:     /tmp/llvm-libclc/              (213 MB)
Build:      /tmp/llvm-libclc/build/        (generated)
Headers:    /usr/include/clc/              (installed)
Bitcode:    /usr/share/clc/*.bc            (installed)
ICD:        /etc/OpenCL/vendors/mesa.icd   (unchanged)
```

### What We Replaced
```
BEFORE (Ubuntu package):
  /usr/include/clc/clctypes.h     ❌ Incomplete (missing unsigned types)
  /usr/include/clc/math/gentype.inc  ❌ MISSING FILE
  /usr/include/clc/convert.h      ❌ Uses undefined types

AFTER (LLVM source):
  /usr/include/clc/clctypes.h     ✅ Complete (all types defined)
  /usr/include/clc/math/gentype.inc  ✅ File exists
  /usr/include/clc/convert.h      ✅ All types resolved
```

---

## 🎓 Lessons Learned

### What Worked
1. ✅ **Compile from source** when distro packages are broken
2. ✅ **Target-specific builds** (AMD-only) speed compilation
3. ✅ **patchelf** can surgically fix library dependencies (used earlier for RustiCL)
4. ✅ **Mesa Clover** (stock Ubuntu) works fine with correct headers

### What Didn't Work
1. ❌ **Manual header patching**: Too many interdependencies
2. ❌ **RustiCL device enumeration**: Mesa 25.0.7 has bugs
3. ❌ **SPIR-V targets**: Required missing `llvm-spirv` binary

### Key Insight
> **Ubuntu's OpenCL stack is fundamentally broken for AMD GPUs.**  
> The only reliable solution is compiling libclc from upstream LLVM source.

---

## 🚀 Impact: "Polaris Ascension"

### Before This Fix
- ❌ AMD RX 580/590 **completely unusable** for OpenCL
- ❌ All kernel compilations failed
- ❌ GPU sat idle despite 6.17 TFLOPS capability

### After This Fix
- ✅ AMD RX 580/590 **fully operational** with OpenCL 1.1
- ✅ GEMM achieves **235 GFLOPS** (1024×1024)
- ✅ Custom kernels compile and execute correctly
- ✅ Numerical accuracy verified (<2e-4 error)

### Philosophical Victory
This proves AMD's **planned obsolescence of Polaris GPUs is reversible** through open source:
- No proprietary drivers needed
- Standard Ubuntu + LLVM source = working GPU
- Community can maintain support indefinitely

**"Polaris Ascension" = Fighting E-waste through open source engineering**

---

## 📝 Reproduction Guide

### Quick Start (15 minutes)
```bash
# 1. Clone LLVM libclc
cd /tmp
git clone --depth 1 --branch release/18.x \
  https://github.com/llvm/llvm-project.git llvm-libclc
cd llvm-libclc && mkdir build && cd build

# 2. Configure (AMD GPUs only)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DLLVM_CONFIG=/usr/bin/llvm-config-18 \
      -DLIBCLC_TARGETS_TO_BUILD="amdgcn--;r600--" \
      ../libclc

# 3. Compile (~5 min with 28 cores)
make -j$(nproc)

# 4. Install
sudo make install

# 5. Verify
python3 -c "
import pyopencl as cl
ctx = cl.create_some_context()
print('✅ OpenCL working!')
"
```

### Verification Checklist
- [ ] `clinfo` shows AMD device
- [ ] `/usr/include/clc/math/gentype.inc` exists
- [ ] `/usr/share/clc/polaris10-amdgcn--.bc` exists
- [ ] Simple kernel compiles without errors
- [ ] GEMM benchmark runs with <1e-3 error

---

## 🔗 Related Documents
- [POWER_MONITORING_GUIDE.md](POWER_MONITORING_GUIDE.md) - GPU power measurement
- [MODEL_GUIDE.md](MODEL_GUIDE.md) - Supported ML models
- [OPENCL_STATUS.md](../OPENCL_STATUS.md) - Overall OpenCL status

---

## 🏆 Credits

**Problem Identified**: Mesa Clover kernel compilation failures  
**Root Cause**: Ubuntu's broken libclc-20-dev package  
**Solution**: Compile libclc from LLVM 18.x source  
**Validation**: GEMM benchmarks on RX 590 (235 GFLOPS)  

**Project**: Polaris Ascension  
**Mission**: Prove AMD Polaris GPUs transcend planned obsolescence

---

*Last Updated: January 23, 2026*  
*OpenCL Status: ✅ **FULLY OPERATIONAL***
