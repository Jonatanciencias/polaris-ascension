# 🧪 Experiments Log

**Research Branch:** Tile=20 Investigation  
**Started:** Febrero 2026

---

## Experiment #1: Approach 1 - Cooperative Loading (v1)

**Date:** 2026-02-04  
**Kernel:** `approach_1_cooperative.cl` (v1)  
**Strategy:** Cooperative loading with modulo indexing

### Results

| Size | Performance | Error | Status |
|------|-------------|-------|--------|
| 512×512 | 631 GFLOPS | 1.19 | ❌ FAIL |
| 1024×1024 | 811 GFLOPS | 2.02 | ❌ FAIL |
| 2048×2048 | 806 GFLOPS | 2.98 | ❌ FAIL |

### Analysis

**Good News:** 🎉
- Performance is EXCELLENT (806 GFLOPS @ 2048)
- 42% improvement over baseline (566 GFLOPS)
- No NaN or Inf values
- Cooperative loading works!

**Problem:** ❌
- Correctness errors too high (2.98 vs. tolerance 0.1)
- Error increases with matrix size
- Likely indexing issue in compute loop

### Root Cause

The compute loop has incorrect indexing:
```c
// Current (WRONG):
if (local_x < TILE_SIZE && local_y < TILE_SIZE) {
    a_val = As[local_x * TILE_SIZE + k];
}
```

Problem: This only accesses first 16×16 of the 20×20 tile!

Threads (16×16) can't directly access all of tile (20×20).

### Next Steps

**Option 1:** Process multiple outputs per thread
- Each thread computes multiple elements
- Better utilization of loaded data

**Option 2:** Use only partial tile
- Load 20×20 but only compute 16×16
- Wasteful but simpler

**Option 3:** Different work-size mapping
- Map global work size to tile size
- Requires different launch configuration

Will try **Option 1** next.

---

## Experiment #2: Approach 1 - Multiple Outputs Per Thread (v2)

**Date:** 2026-02-04  
**Status:** 📝 PLANNED

### Strategy

Instead of each thread computing 1 row × 4 columns, compute multiple rows:
- Thread (local_x, local_y) computes rows: local_x, local_x+16, ...
- Full utilization of 20×20 tile
- More work per thread

**Expected:** Higher correctness, similar performance

---

_Log continues as experiments progress..._
