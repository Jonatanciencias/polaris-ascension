# tile28/tile32 Feasibility Analysis

**Date**: 5 de febrero de 2026  
**Context**: Evaluating whether larger tiles could improve performance for matrices > 3072  
**Current best**: 805 GFLOPS @ 3072 (tile24)

---

## 🎯 Hypothesis

**Claim**: tile28 or tile32 might achieve 850+ GFLOPS on matrices 4096+

**Rationale**:
- tile24 @ 3072: 805 GFLOPS
- Larger tiles = more work per thread = better compute/memory ratio
- Potentially better cache utilization for very large matrices

---

## 🔍 Hardware Constraints Analysis

### RX 590 GME (GCN 4.0 Polaris) Limits

| Resource | Limit | tile24 (current) | tile28 | tile32 |
|----------|-------|------------------|--------|--------|
| **Max workgroup size** | 256 threads | 144 (57%) | 196 (76%) | 256 (100%) ⚠️ |
| **Local memory** | 32 KB | 4.6 KB (14%) | 6.3 KB (20%) | 8.2 KB (26%) |
| **Registers per CU** | 65,536 | Moderate | High ⚠️ | Very High ⚠️ |
| **Wavefront size** | 64 | 2.25 waves | 3.06 waves | 4 waves |

### tile28 Analysis

**Configuration**:
- Workgroup: 14×14 = 196 threads
- Tile: 28×28 elements
- LDS: 2 × 28×28 × 4 = 6,272 bytes

**Pros**:
- ✅ Within all hardware limits
- ✅ 3.06 wavefronts (decent occupancy)
- ✅ Moderate LDS usage

**Cons**:
- ❌ No perfect alignment: 4096 / 28 = 146.3 tiles (requires padding)
- ⚠️ Increased register pressure vs tile24
- ⚠️ No clear advantage (odd tile size, no power of 2)

**Verdict**: ⭐⭐ UNLIKELY to beat tile24 (no alignment benefit, higher complexity)

---

### tile32 Analysis

**Configuration**:
- Workgroup: 16×16 = 256 threads (MAXIMUM)
- Tile: 32×32 elements
- LDS: 2 × 32×32 × 4 = 8,192 bytes

**Pros**:
- ✅ Perfect alignment: 4096 / 32 = 128 tiles (no padding!)
- ✅ Power of 2 (compiler optimizations)
- ✅ Larger tiles = fewer kernel launches

**Cons**:
- ❌ **MAX workgroup size** (256 threads) - reduces occupancy
- ❌ **High register pressure** - each thread processes 2×2 = 4 outputs
  - Accumulator: 4 floats
  - Tile indices: multiple
  - Loop variables: multiple
  - **Risk**: Register spilling (like float8 experiment)
- ❌ 4 full wavefronts = may limit parallel execution

**Verdict**: ⭐⭐⭐ POSSIBLE but HIGH RISK (perfect alignment vs register spillage)

---

## 📊 Expected Performance Analysis

### Theoretical Model

**Memory bandwidth bound check**:
- RX 590: 256 GB/s theoretical
- 4096×4096 GEMM: 3 matrices × 4096² × 4 bytes = 201 MB
- Bandwidth used: ~160-180 GB/s @ 805 GFLOPS
- **Still memory-bound, not compute-bound**

**Impact of larger tiles**:
- tile24 → tile28: +640 bytes LDS per workgroup (+37%)
- tile24 → tile32: +3584 bytes LDS per workgroup (+78%)
- **More LDS = fewer workgroups in flight = WORSE occupancy**

### Pessimistic Case (Register Spilling)
```
tile32 spills registers → -50-60% performance
Result: 805 → 350-400 GFLOPS
Like float8 experiment (-60%)
```

### Optimistic Case (No Spilling, Perfect Alignment)
```
tile32 @ 4096: Perfect alignment benefit
Estimate: +5-8% over tile24
Result: 805 → 845-870 GFLOPS
```

### Realistic Case
```
Slight occupancy reduction, moderate register pressure
tile32 @ 4096: ±0-5%
Result: 805 → 805-845 GFLOPS (marginal improvement)
```

---

## 🧪 Risk-Benefit Analysis

### tile28
- **Time**: 3-4 hours
- **Probability of success**: 20-30% (no clear advantage)
- **Expected gain**: 0-20 GFLOPS (+0-2%)
- **ROI**: ⭐⭐ POOR (effort >> potential gain)
- **Recommendation**: **SKIP** (not worth it)

### tile32
- **Time**: 3-4 hours
- **Probability of success**: 40-50% (alignment helps, but spillage risk)
- **Expected gain**: Best case +45 GFLOPS (+5.6%), Realistic +0-40 GFLOPS
- **Downside risk**: -50% (like float8)
- **ROI**: ⭐⭐⭐ MODERATE (interesting experiment, but uncertain)
- **Recommendation**: **CONDITIONAL** (only if you NEED 4096+ matrices)

---

## 🎯 Decision Tree

### Question 1: Do you actually use 4096×4096 matrices?

**NO** → **SKIP tile32**
- Current 805 GFLOPS @ 3072 is excellent
- No real-world benefit
- Academic curiosity only

**YES** → Continue to Question 2

### Question 2: Can you afford the risk?

**NO** (need stability) → **SKIP tile32**
- Might lose performance (spillage)
- Already have working system
- Not worth destabilizing

**YES** (learning/curiosity) → **PROCEED with tile32 ONLY**
- Skip tile28 (no advantage)
- Implementation plan below

---

## 📋 Implementation Plan (If Proceeding)

### Phase 1: Minimal Viable tile32 (2 hours)

```c
// Copy tile24, adjust to tile32
#define TILE_SIZE 32
#define LOCAL_SIZE 16  // 16×16 = 256 threads

__kernel void gemm_tile32_experimental(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    __local float As[32][32];
    __local float Bs[32][32];
    
    // Each thread computes 2×2 output elements
    // ... (similar to tile24)
}
```

### Phase 2: Validation (30 min)

```bash
# Test on multiple sizes
python benchmark_tile32.py --sizes 2048,3072,4096,5120

# Check correctness
assert max_error < 0.001

# Compare vs tile24
if tile32_4096 > tile24_4096 + 20:
    print("SUCCESS: tile32 wins on large matrices")
elif tile32_4096 < tile24_4096 - 50:
    print("FAILED: Register spilling confirmed")
else:
    print("MARGINAL: Not worth the complexity")
```

### Phase 3: Decision (30 min)

**If tile32 wins (+20 GFLOPS or more)**:
- Integrate into selector
- Update documentation
- Add to production

**If tile32 fails or marginal**:
- Document why it failed
- Add to FAILED_EXPERIMENTS.md
- Learn from experience

---

## 🔬 Alternative: Investigate Current tile24 @ 4096

### Before trying tile32, benchmark tile24 @ 4096

```bash
python -c "
from src.optimization_engines.adaptive_kernel_selector import select_optimal_kernel
import pyopencl as cl
import numpy as np

# Test 4096×4096 with tile24
M, N, K = 4096, 4096, 4096
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Run benchmark
# ... (benchmark code)

print(f'tile24 @ 4096: {gflops:.1f} GFLOPS')
"
```

**If tile24 @ 4096 is already 800+ GFLOPS**:
- tile32 won't help much (alignment matters less than expected)
- SKIP the experiment

**If tile24 @ 4096 drops significantly (< 750 GFLOPS)**:
- tile32 might help (alignment could be the issue)
- Worth testing

---

## 💡 MY RECOMMENDATION

### Option A: Quick Test First (10 minutes) ⭐⭐⭐⭐⭐

```bash
# Test tile24 @ 4096 first
cd /path/to/project
python -c "
import sys
sys.path.insert(0, 'src')
from optimization_engines.adaptive_kernel_selector import ProductionKernelSelector
import time

selector = ProductionKernelSelector()

# Quick test
rec = selector.select_kernel(4096, 4096, 4096)
print(f'Recommended for 4096: {rec[\"kernel_key\"]} - {rec[\"predicted_gflops\"]:.1f} GFLOPS')

# Run actual benchmark if you have the infrastructure
"
```

**Decision based on result**:
- If **800+ GFLOPS**: SKIP tile32 (tile24 already excellent)
- If **< 750 GFLOPS**: Consider tile32 (room for improvement)
- If **750-800 GFLOPS**: Marginal case (your call)

### Option B: Skip Entirely ⭐⭐⭐⭐ (RECOMMENDED)

**Rationale**:
1. tile24 @ 3072 already hits 805 GFLOPS
2. Going beyond 3072 has diminishing returns (memory bound)
3. Register spilling risk is real (float8 precedent)
4. tile32 alignment benefit might be theoretical only
5. **You already have +42% improvement** - excellent result

**Alternative use of 3-4 hours**:
- Write blog post about the journey
- Create comparison with CLBlast
- Develop educational content
- Test on different AMD GPUs (community)

### Option C: Document Why NOT to Try ⭐⭐⭐⭐⭐ (BEST)

**Create**: `TILE32_ANALYSIS_SKIP_DECISION.md`

**Content**:
- Hardware constraint analysis (this document)
- Risk-benefit calculation
- Decision to SKIP based on:
  - High risk of register spilling
  - Marginal expected benefit
  - No real-world use case for 4096+ on RX 590
  - Better ROI in other activities

**Value**: Shows professional decision-making (knowing when NOT to optimize)

---

## 📊 Summary Table

| Option | Time | Risk | Expected Gain | ROI | Recommendation |
|--------|------|------|---------------|-----|----------------|
| **tile28** | 3-4h | Medium | 0-20 GFLOPS (+0-2%) | ⭐⭐ POOR | ❌ SKIP |
| **tile32** | 3-4h | High | 0-45 GFLOPS (+0-5%) | ⭐⭐⭐ MODERATE | ⚠️ CONDITIONAL |
| **Quick test tile24@4096** | 10min | None | Insight | ⭐⭐⭐⭐⭐ EXCELLENT | ✅ DO THIS |
| **Skip & document** | 30min | None | Professional decision | ⭐⭐⭐⭐⭐ EXCELLENT | ✅ RECOMMENDED |
| **Alternative: Publish** | 2-4h | None | Community impact | ⭐⭐⭐⭐⭐ EXCELLENT | ✅ BEST |

---

## 🎯 FINAL RECOMMENDATION

### DO THIS (10 minutes):

```bash
# Quick benchmark tile24 @ 4096
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580
python3 -c "
import sys
sys.path.insert(0, 'src')
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

# Load tile24 kernel
with open('src/kernels/gemm_tile24_production.cl') as f:
    prg = cl.Program(ctx, f.read()).build()

# Test 4096
M = N = K = 4096
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

# ... benchmark code ...

print(f'Result: tile24 @ 4096 = {gflops:.1f} GFLOPS')
"
```

### THEN DECIDE:

**If 800+ GFLOPS**: 
✅ **STOP HERE** - tile24 is excellent, no need for tile32
→ Proceed to publication

**If < 750 GFLOPS**: 
⚠️ **Consider tile32** (but understand the risks)
→ Worth the 3-4 hour experiment

**My prediction**: tile24 @ 4096 will be **780-810 GFLOPS** (excellent)  
**Therefore**: Skip tile32, publish current results

---

## 📝 Conclusion

**Bottom line**: tile32 is **risky** with **uncertain payoff**

**Smarter move**: Test tile24 @ 4096 (10 min), then decide

**Best move**: Skip optimization, focus on **community impact** (publish, share, help others)

**You already have**: 805-810 GFLOPS (+42%) - this is **publication-ready**! 🎉
