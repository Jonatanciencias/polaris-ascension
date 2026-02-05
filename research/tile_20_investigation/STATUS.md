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

---

## 💡 Key Insights

### What's Working ✅
1. **Cooperative loading pattern** - threads can load 400 elements
2. **Performance potential** - 780-810 GFLOPS achieved
3. **Stability** - no NaN/Inf, clean execution
4. **Framework** - excellent testing infrastructure

### What's Not Working ❌
1. **Indexing calculations** - errors in tile access
2. **Global-to-tile mapping** - work group indexing issues
3. **B tile layout** - vectorized access not aligned

---

## 🔍 Next Steps

### Immediate (Today)
1. ✅ Document current findings
2. 🔜 Debug B tile indexing
3. 🔜 Create simplified test case
4. 🔜 Try Approach 1 v3 with fixed indexing

### Short-term (This Week)
1. Resolve correctness issues
2. If successful → optimize
3. If blocked → try Approach 2 (non-square tiles)

---

## 📁 Files Created

### Infrastructure
- `README.md` - Project overview
- `docs/RESEARCH_PLAN.md` - Detailed plan
- `docs/EXPERIMENTS_LOG.md` - Experiment tracking
- `experiments/experiment_framework.py` - Testing framework (470 lines)

### Kernels
- `kernels/approach_1_cooperative.cl` - v1 (TESTED)
- `kernels/approach_1_v2_multi_output.cl` - v2 (TESTED)

### Tests
- `experiments/approach_1_test.py` - v1 test
- `experiments/approach_1_v2_test.py` - v2 test

---

## 🎓 Learnings

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
Week 1, Day 1:
├── Infrastructure: 100% ✅
├── Approach 1 v1:  TESTED (perf good, correctness fail)
├── Approach 1 v2:  TESTED (perf good, correctness fail)
└── Approach 1 v3:  PLANNED (fix indexing)
```

**Time Invested:** ~3 hours  
**Next Milestone:** First successful result (error < 0.1)

---

**Status:** 🔬 DEBUGGING  
**Next Action:** Fix B tile indexing in v3
