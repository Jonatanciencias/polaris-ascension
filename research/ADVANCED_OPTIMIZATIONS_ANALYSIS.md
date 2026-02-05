# Advanced Optimizations Analysis - Rectangular Tiles, Fusion, Batching

**Date**: 5 de febrero de 2026  
**Context**: Post tile32 decision, evaluating remaining optimization opportunities  
**Current Status**: 805-810 GFLOPS achieved, +42-43% vs baseline

---

## 🎯 Overview

We have three potential optimization directions to evaluate:
1. **Rectangular Tiles** (non-square tiles like 20×24, 16×32)
2. **Kernel Fusion** (GEMM+activation in single kernel)
3. **Batched GEMM** (multiple small matrices in parallel)

Each requires different analysis methodology.

---

## 1️⃣ RECTANGULAR TILES (20×24, 16×32)

### Concept

**Current**: Square tiles (20×20, 24×24)  
**Proposal**: Rectangular tiles to match non-square matrices

**Example**:
```
Matrix A: 1400×2048 (M×K)
Matrix B: 2048×1400 (K×N)
Result C: 1400×1400 (M×N)

Current approach:
- tile20: Works on 1400×1400 output (square)
- But input K=2048 is different (rectangular data)

Rectangular tile idea:
- tile20×24: 20 rows (M), 24 cols (N) per workgroup
- Better match for rectangular geometries?
```

### When It Might Help

**Scenario A: Non-square output matrices**
```python
# Example: Neural network layer
M, N, K = 1400, 2048, 1536  # C is 1400×2048 (rectangular!)

# tile20 (square): Processes 20×20 chunks → inefficient at boundaries
# tile20×24: Could process 20×24 chunks → better boundary handling
```

**Scenario B: Non-square memory access patterns**
```python
# Different memory coalescing in M vs N dimensions
# Rectangular tile could optimize for asymmetric bandwidth
```

### Analysis

**Pros**:
- ✅ Could optimize boundary handling for non-square matrices
- ✅ Flexibility in workgroup distribution
- ✅ Might align better with cache geometries

**Cons**:
- ❌ **Most GEMM is square or near-square** (ML: 1024×1024, 2048×2048)
- ❌ Doubles kernel count (need tile20x24, tile24x20, tile16x32, tile32x16...)
- ❌ Complex ML selector retraining (13 features → 20+ features)
- ❌ No clear theoretical advantage (GPU is symmetric in X/Y)
- ❌ Maintenance burden (4-8 more kernels to test/validate)

**ROI**: ⭐⭐ POOR
- Expected gain: +0-5% on non-square matrices only
- Effort: 10-15 hours (multiple kernels + selector + validation)
- Complexity increase: HIGH

### Quick Test

```bash
# Before implementing, profile actual workload:
python -c "
import numpy as np
# Check your actual matrix sizes
workload = [
    (1024, 1024, 1024),  # square
    (2048, 2048, 2048),  # square
    (1400, 1400, 1400),  # square
    # Are any of your matrices actually rectangular?
]
"
```

**Recommendation**: ❌ **SKIP**
- Real-world workloads are predominantly square
- High complexity, low expected benefit
- Not worth 10-15 hours investment

---

## 2️⃣ KERNEL FUSION (GEMM+ReLU+Bias)

### Concept

**Traditional approach** (3 kernel launches):
```python
# Pass 1: GEMM
C = matmul(A, B)  # 805 GFLOPS, memory write

# Pass 2: Add bias
C = C + bias      # Memory read + write

# Pass 3: ReLU activation
C = max(0, C)     # Memory read + write

# Total: 1 compute + 4 memory operations
```

**Fused approach** (1 kernel launch):
```c
__kernel void gemm_relu_bias_fused(...) {
    // ... GEMM computation ...
    
    // Immediately apply bias + ReLU before writing
    float result = acc[i][j];
    result = result + bias[col];      // Add bias
    result = fmax(0.0f, result);      // ReLU
    C[row * N + col] = result;        // Write once
}

// Total: 1 compute + 1 memory write (4× reduction!)
```

### When It Helps DRAMATICALLY

**ML Inference Pipeline**:
```python
# Typical transformer layer
x = matmul(input, W1)     # 805 GFLOPS
x = x + bias              # Memory bound
x = relu(x)               # Memory bound
x = matmul(x, W2)         # 805 GFLOPS
# ... repeat ...

# With fusion:
x = fused_gemm_relu_bias(input, W1, bias)  # Same 805 GFLOPS
x = matmul(x, W2)         # 805 GFLOPS
# ... 30-40% faster end-to-end!
```

### Analysis

**Pros**:
- ✅ **Huge memory traffic reduction** (4× fewer memory ops)
- ✅ **20-40% end-to-end speedup** in ML pipelines
- ✅ Common pattern (every linear layer in neural nets)
- ✅ Better cache utilization
- ✅ Reduced kernel launch overhead

**Cons**:
- ❌ **Not general-purpose** (specific to ML use case)
- ❌ Need multiple variants (ReLU, GELU, Tanh, Sigmoid, etc.)
- ❌ Different API than standard GEMM
- ❌ Complicates testing (more combinations)
- ⚠️ Same GEMM performance (805 GFLOPS), not faster

**Effort**: 6-10 hours
- Modify tile20 kernel: 2 hours
- Add bias+activation: 1 hour
- Testing: 2-3 hours
- Integration: 2-3 hours
- Documentation: 1 hour

**Expected Benefit**:
- GEMM alone: 0 improvement (still 805 GFLOPS)
- **Pipeline end-to-end: +20-40%** (memory savings dominate)

**ROI**: ⭐⭐⭐⭐ **VERY GOOD for ML use cases**

### Quick Test

**Do you use this pattern?**
```python
# If your code looks like this:
C = torch.matmul(A, B)
C = C + bias
C = torch.relu(C)

# Or this:
from torch.nn import Linear, ReLU

layer = Linear(1024, 2048)
activation = ReLU()

out = layer(x)
out = activation(out)

# Then YES, fusion helps!
```

**Recommendation**: ⚠️ **CONDITIONAL**
- **IF** you're integrating into PyTorch/TensorFlow: ✅ **DO IT**
- **IF** standalone GEMM library: ❌ **SKIP** (wrong focus)
- **IF** building ML inference engine: ✅ **HIGH PRIORITY**

**Priority**: After publication, if building ML integration

---

## 3️⃣ BATCHED GEMM (100× 256×256 matrices)

### Concept

**Traditional approach**:
```python
# Process 100 small matrices one by one
results = []
for i in range(100):
    C_i = matmul(A_i, B_i)  # 256×256
    results.append(C_i)

# 100 kernel launches
# High overhead (each launch: 5-10 μs)
# Total overhead: 0.5-1 ms
```

**Batched approach**:
```c
__kernel void gemm_batched(
    const int num_matrices,  // 100
    const int M, const int N, const int K,  // 256, 256, 256
    __global const float* A_batch,  // All A matrices contiguous
    __global const float* B_batch,  // All B matrices contiguous
    __global float* C_batch
) {
    // Get batch index from workgroup ID
    int batch_idx = get_group_id(2);  // 3D grid!
    
    // Offset pointers
    __global const float* A = A_batch + batch_idx * M * K;
    __global const float* B = B_batch + batch_idx * K * N;
    __global float* C = C_batch + batch_idx * M * N;
    
    // Regular GEMM on this batch element
    // ... same as tile20/tile24 ...
}

// 1 kernel launch for all 100 matrices!
```

### When It Helps DRAMATICALLY

**Transformer Models**:
```python
# Multi-head attention (8 heads, batch size 16)
# = 128 small matrix multiplications

# Traditional: 128 launches → 1.28 ms overhead
# Batched: 1 launch → 0.01 ms overhead
# Speedup: 2-3× on small matrices!
```

**3D Convolution** (depthwise):
```python
# 64 channels, each does small GEMM
# Traditional: 64 launches
# Batched: 1 launch
```

### Analysis

**Pros**:
- ✅ **2-3× speedup on small matrices** (< 512×512)
- ✅ Dramatically reduces kernel launch overhead
- ✅ Common in modern ML (attention mechanisms)
- ✅ Better GPU utilization (more work in flight)
- ✅ Scales with batch size (100 matrices → 100× parallel)

**Cons**:
- ❌ **Only helps for small matrices** (< 512×512)
- ❌ Requires 3D workgroup dispatch (more complex)
- ❌ Memory layout must be contiguous (batch-first)
- ❌ Different API (not drop-in replacement)
- ⚠️ RX 590 has 36 CUs → can process ~18-36 small matrices in parallel
  - 100 matrices needs sequential waves anyway

**Effort**: 8-12 hours
- Design API: 2 hours
- Implement batched kernel: 3 hours
- Test various batch sizes: 3 hours
- Integration + documentation: 3-4 hours

**Expected Benefit**:
- Small matrices (256×256): **2-3× throughput** vs sequential
- Medium matrices (512×512): **1.5-2× throughput**
- Large matrices (1024+): **No benefit** (already saturating GPU)

**ROI**: ⭐⭐⭐⭐ **VERY GOOD for ML batch inference**

### Quick Test

**Profile your workload**:
```bash
# Do you have this pattern?
python -c "
# Option 1: Explicit batch
batch_size = 16
seq_len = 128
hidden = 256

# Many small GEMMs?
for b in range(batch_size):
    for head in range(8):  # Multi-head attention
        Q @ K.T  # 128×256 @ 256×128 → 128×128 (small!)
        
# Option 2: Single large GEMM
Q @ K.T  # (16*128)×256 @ 256×(128) → 2048×128 (large, already fast)
"
```

**Reality Check**:
- Modern frameworks (PyTorch/TF) already batch automatically
- You only need this if building custom inference engine
- Or if framework can't batch (rare specialized cases)

**Recommendation**: ⚠️ **CONDITIONAL**
- **IF** building custom ML inference: ✅ **HIGH VALUE**
- **IF** using PyTorch/TensorFlow: ❌ **SKIP** (already batched)
- **IF** standalone GEMM library: ❌ **SKIP** (wrong focus)

**Priority**: High for custom inference, low for general library

---

## 📊 COMPARISON TABLE

| Optimization | Effort | Complexity | Expected Gain | Use Case Specificity | ROI | Recommendation |
|--------------|--------|------------|---------------|---------------------|-----|----------------|
| **Rectangular Tiles** | 10-15h | High | +0-5% | General (rare) | ⭐⭐ POOR | ❌ SKIP |
| **Kernel Fusion** | 6-10h | Medium | +20-40% e2e | ML pipelines | ⭐⭐⭐⭐ GOOD | ✅ IF ML integration |
| **Batched GEMM** | 8-12h | Medium-High | 2-3× on small | ML batch inference | ⭐⭐⭐⭐ GOOD | ✅ IF custom engine |

---

## 🎯 DECISION FRAMEWORK

### Question 1: What's your primary use case?

**A. General-purpose GEMM library** (like CLBlast)
- Skip all three ❌
- Focus on: Publication, benchmarking, community
- **Reason**: These are application-specific optimizations

**B. PyTorch/TensorFlow custom op**
- Kernel fusion: ✅ YES (high impact)
- Batched GEMM: ❌ SKIP (framework already does it)
- Rectangular: ❌ SKIP

**C. Custom ML inference engine** (from scratch)
- Kernel fusion: ✅ YES (mandatory)
- Batched GEMM: ✅ YES (mandatory)
- Rectangular: ❌ SKIP
- **Note**: Building full inference engine = 100+ hours

**D. Research/learning project**
- All three: ❌ Better to publish and move on
- **Reason**: Diminishing returns, publication has higher impact

### Question 2: What's your timeline?

**This week**: ❌ Skip all, focus on publication
**This month**: ⚠️ Maybe fusion (if ML use case)
**This quarter**: ✅ Plan inference engine (includes fusion + batching)

### Question 3: What's your goal?

**Maximum GEMM performance**: ❌ Already achieved (805 GFLOPS)
**ML inference performance**: ✅ Fusion + batching
**Learning experience**: ⚠️ Pick one (fusion is cleanest)
**Community impact**: ❌ Publish current work instead

---

## 💡 MY RECOMMENDATION

### For Your Current Project: ❌ **SKIP ALL THREE**

**Reasons**:
1. **You have a general-purpose GEMM library** (805 GFLOPS, excellent)
2. **These are application-specific optimizations**
   - Fusion: Only helps in ML pipelines
   - Batching: Only helps for small matrix batches
   - Rectangular: Rarely needed in practice
3. **Higher ROI activities**:
   - Publish blog post (community impact)
   - GitHub release (visibility)
   - Benchmark vs CLBlast (credibility)
   - Support questions from users (community building)

### If You Want to Continue: 🎯 **CHOOSE A DIRECTION**

**Option A: ML Inference Stack** (3-6 months)
```
Phase 1 (Done): Core GEMM (805 GFLOPS) ✅
Phase 2 (2-3 weeks): Kernel fusion variants (ReLU, GELU, etc.)
Phase 3 (2-3 weeks): Batched GEMM
Phase 4 (1-2 months): Conv2D, attention, layer norm
Phase 5 (1 month): PyTorch integration

Result: Complete inference library for RX 590
Impact: High (can run models locally)
```

**Option B: General GEMM Library** (current)
```
Phase 1 (Done): Core GEMM (805 GFLOPS) ✅
Phase 2 (Current): Publication & community
Phase 3 (1-2 weeks): Benchmarking vs CLBlast
Phase 4 (Ongoing): Community support, PRs
Phase 5 (Future): Port to other AMD GPUs

Result: Best-in-class GEMM for Polaris
Impact: High (helps AMD GPU compute ecosystem)
```

**My vote**: Option B (publish current, don't add complexity)

---

## 📝 SUMMARY

### TL;DR

**Rectangular Tiles**: ❌ Skip (low value, high complexity)
**Kernel Fusion**: ⚠️ Only if building ML pipeline (20-40% e2e gain)
**Batched GEMM**: ⚠️ Only if building custom inference (2-3× on small matrices)

**For your current project (general GEMM)**: ❌ **SKIP ALL THREE**

**Better use of time**:
1. **This week**: Publish blog post + GitHub release
2. **Next week**: Benchmark vs CLBlast, share results
3. **Next month**: Support community, accept PRs, extend to Vega/RDNA

**Why**: You have excellent standalone GEMM (805 GFLOPS). These optimizations are for **integrated pipelines**, not **library functions**. Different goals, different trade-offs.

---

## 🎯 FINAL ANSWER

**Should you implement these three?**

**Short answer**: NO

**Medium answer**: Not now, maybe later if you pivot to ML inference

**Long answer**: You've built an excellent general-purpose GEMM library. These optimizations are application-specific (ML pipelines). If you want maximum impact, **publish what you have** (805 GFLOPS standalone GEMM is publication-worthy). If you want to pivot to ML inference, **start a new project** (these three are just the beginning of a 6-month inference engine project).

**Don't fall into the trap**: "Just one more optimization..." 

**You're done with GEMM optimization.** 🎉 

**Next phase**: **SHARE IT WITH THE WORLD** 🚀

---

**Status**: ✅ Analysis complete  
**Recommendation**: Skip all three for current project  
**Next action**: Proceed to publication  
**Alternative path**: Start ML inference project (separate effort)
