# 📊 Tile=20 Investigation - Current Status

**Last Updated:** 2026-02-04  
**Status:** 🔬 ACTIVE RESEARCH - Week 1, Day 1

---

## 🎯 Quick Summary

**Research Goal:** Achieve ≥900 GFLOPS by integrating tile=20 (from auto-tuner's 1148 GFLOPS discovery)

**Current Status:**
- ✅ Research infrastructure complete
- 🔬 Testing Approach 1 (variations)
- ⚠️ Performance good (784 GFLOPS), correctness issues persist

---

## 📊 Results So Far

### Approach 1 v1: Cooperative Loading
- **Performance:** 806 GFLOPS @ 2048 (+42% vs. 566 baseline)
- **Correctness:** ❌ FAIL (max_error=2.98)
- **Issue:** Only used 16×16 of loaded 20×20 tile

### Approach 1 v2: Multiple Outputs Per Thread
- **Performance:** 784 GFLOPS @ 2048 (+39% vs. 566 baseline)
- **Correctness:** ❌ FAIL (max_error=2.18)
- **Issue:** Indexing error in B tile access

### Approach 1 v3: Fixed Indexing
- **Performance:** 724 GFLOPS @ 2048 (+28% vs. 566 baseline)
- **Correctness:** ❌ FAIL (max_error=2.16)
- **Issue:** Fundamental thread-to-output mismatch (256 ≠ 400)

### Approach 2: Non-Square (10×10 threads)
- **Performance:** 554 GFLOPS @ 1024 (-2% vs. 566 baseline)
- **Correctness:** ✅ **PASS** (max_error=0.000002) 🎉
- **Issue:** Only 100 threads, low occupancy

**KEY INSIGHT:** Correctness achieved! Need to optimize occupancy.

---

## 💡 Key Insights

### What's Working ✅
1. **Cooperative loading pattern** - threads can load 400 elements
2. **Performance potential** - 780-810 GFLOPS achieved (though incorrect)
3. **Stability** - no NaN/Inf, clean execution
4. **Framework** - excellent testing infrastructure
5. **Approach 2: CORRECTNESS!** - 100% correct with 10×10 threads ✅

### What's Not Working ❌
1. **Approach 1:** 256 threads don't map to 400 outputs cleanly
2. **Approach 2:** Only 100 threads = low occupancy = slower than baseline

---

## 🔍 Next Steps

### Immediate (Today)
1. ✅ Document current findings
2. ✅ Debug indexing → Found fundamental issue (256 ≠ 400)
3. ✅ Try Approach 2 → **SUCCESS!** (correct but slow)
4. 🔜 Optimize Approach 2 for higher occupancy

### Short-term (This Week)
1. Try 20×10 threads (200) or 20×12 threads (240)
2. Optimize Approach 2 for performance
3. Compare all approaches
4. Decision: integrate, optimize more, or archive

---

## 📁 Files Created

### Infrastructure
- `README.md` - Project overview
- `docs/RESEARCH_PLAN.md` - Detailed plan
- `docs/EXPERIMENTS_LOG.md` - Experiment tracking
- `experiments/experiment_framework.py` - Testing framework (470 lines)

### Kernels, perf good, incorrect)
- `kernels/approach_1_v2_multi_output.cl` - v2 (TESTED, perf good, incorrect)
- `kernels/approach_1_v3_fixed_indexing.cl` - v3 (TESTED, perf ok, incorrect)
- `kernels/approach_2_nonsquare.cl` - **10×10 threads** (TESTED, ✅ CORRECT!) 🎉

### Tests
- `experiments/approach_1_test.py` - v1 test
- `experiments/approach_1_v2_test.py` - v2 test
- `experiments/approach_1_v3_test.py` - v3 test
- `experiments/approach_2_test.py` - **FIRST SUCCESS!** ✅
- `experiments/approach_1_v2_test.py` - v2 test

---
2. ✅ **Correctness IS achievable** - Approach 2 proves it with 10×10 threads
3. ⚠️ **256 threads ≠ 400 outputs** - Fundamental mismatch causes errors
4. ✅ **Simple mapping wins** - 100 threads × 4 outputs = clean, correct
5. ⚠️ **Occupancy matters** - 100 threads too few, need 200-256 for speed
6. ✅ **Framework is valuable** - systematic testing revealing insights
1. **Cooperative loading works** - 256 threads CAN load 400 elements efficiently
2. **Performance is there** - 780-810 GFLOPS proves tile=20 potential
3. **Indexing is complex** - need careful work-group to tile mapping
4. **Framework is valuable** - systematic testing catching issues early

---

## 🛡️ Production Safety

✅ **Production code UNTOUCHED**
- No changes to `src/`
- 566 GFLOPS baseline still working
- Can revert research at any time

---

## 📈 Progress

```
Week 1, Day 1 (4-6 hours invested):
├── Infrastructure: 100% ✅
├── Approach 1 v1:  TESTED (806 GFLOPS, incorrect)
├── Approach 1 v2:  TESTED (784 GFLOPS, incorrect)
├── Approach 1 v3:  TESTED (724 GFLOPS, incorrect)
├── Approach 2:     TESTED (554 GFLOPS, ✅ CORRECT!) 🎉
└── Approach 2 v2:  PLANNED (optimize occupancy)
```

**Time Invested:** ~6 hours  
**Next Milestone:** Optimize Approach 2 to beat baseline (>566 GFLOPS)

---🎉 **FIRST SUCCESS!**  
**Next Action:** Optimize Approach 2 with higher thread count (20×10 or 20×12)
**Status:** 🔬 DEBUGGING  
**Next Action:** Fix B tile indexing in v3
