# 🔬 Tile=20 Investigation - Research Branch

**Created:** Febrero 2026  
**Status:** 🧪 EXPERIMENTAL  
**Goal:** Achieve 1148 GFLOPS by integrating tile=20 configuration

---

## 🎯 Objective

Investigate and implement the auto-tuner's best configuration (T20_L16x16_U4 = 1148 GFLOPS) while maintaining production code stability.

**Production Code:** UNTOUCHED ✅ (566 GFLOPS, stable)  
**Research Code:** EXPERIMENTAL 🔬 (this directory)

---

## 📊 Current Situation

### Production Performance (STABLE)
- **Kernel:** FLOAT4_VEC (tile=16)
- **Performance:** 566 GFLOPS @ 2048×2048
- **Correctness:** 100% (max_error < 0.001)
- **Status:** Production-ready ✅

### Auto-Tuner Discovery (POTENTIAL)
- **Configuration:** T20_L16x16_U4
- **Standalone Performance:** 1148 GFLOPS (+102%)
- **Challenge:** Integration issues with 256-thread limit

---

## 🧩 The Integration Challenge

### Root Problem

```
Hardware Constraint:
  AMD RX 590 GME max work group size = 256 threads
  
Current Production (tile=16):
  local_size (16,16) = 256 threads
  Tile 16×16         = 256 elements
  Coverage           = 100% ✅ PERFECT FIT
  
Research Target (tile=20):
  local_size (16,16) = 256 threads
  Tile 20×20         = 400 elements
  Coverage           = 64% ⚠️ INSUFFICIENT
```

### Previous Integration Attempts

**Attempt #1: Direct Integration**
- Result: 1169 GFLOPS but NaN errors
- Issue: Threads insufficient to properly load tile

**Attempt #2: Cooperative Loading**
- Result: 674 GFLOPS with large errors
- Issue: Compute loop indexing mismatch

---

## 🔬 Research Approaches

### Approach 1: Cooperative Loading with Fixed Indexing

**Idea:** Separate loading phase from compute phase
- **Loading:** Cooperative (256 threads load 400 elements)
- **Compute:** Use modulo arithmetic for proper indexing
- **Sync:** Barrier between phases

**Implementation:** `experiments/approach_1_cooperative_fixed.py`

---

### Approach 2: Non-Square Tiles

**Idea:** Use tiles that fit thread geometry better
- **Candidates:** 16×20, 20×16, 16×24, 24×16
- **Advantage:** Better thread-to-element mapping
- **Trade-off:** May need asymmetric loading

**Implementation:** `experiments/approach_2_nonsquare_tiles.py`

---

### Approach 3: Transposed Tile Layout

**Idea:** Transpose one tile for better memory access
- **Layout:** A tiles normal, B tiles transposed
- **Advantage:** May reduce bank conflicts
- **Trade-off:** More complex loading pattern

**Implementation:** `experiments/approach_3_transposed_tiles.py`

---

### Approach 4: Hierarchical Tiling

**Idea:** Break 20×20 into smaller sub-tiles
- **Strategy:** Process 4 sub-tiles of 10×10
- **Advantage:** Fits within thread limits
- **Trade-off:** More synchronization overhead

**Implementation:** `experiments/approach_4_hierarchical.py`

---

### Approach 5: Multiple Work-Items Per Result

**Idea:** Use reduction within work-group
- **Strategy:** Multiple threads compute partial sums
- **Advantage:** Natural fit for thread count
- **Trade-off:** Reduction overhead

**Implementation:** `experiments/approach_5_reduction.py`

---

### Approach 6: Hybrid Tile Sizes

**Idea:** Use tile=20 for large matrices, tile=16 for medium
- **Strategy:** Dynamic selection based on size
- **Advantage:** Best of both worlds
- **Trade-off:** More kernel variants

**Implementation:** `experiments/approach_6_hybrid.py`

---

## 📁 Directory Structure

```
research/tile_20_investigation/
├── README.md                    # This file
├── docs/
│   ├── RESEARCH_PLAN.md        # Detailed research plan
│   ├── EXPERIMENTS_LOG.md      # Experiment results
│   └── FINDINGS.md             # Key findings and insights
├── kernels/
│   ├── base_tile20.cl          # Base tile=20 kernel
│   ├── approach_1.cl           # Cooperative loading
│   ├── approach_2.cl           # Non-square tiles
│   ├── approach_3.cl           # Transposed layout
│   ├── approach_4.cl           # Hierarchical tiling
│   ├── approach_5.cl           # Reduction-based
│   └── approach_6.cl           # Hybrid approach
└── experiments/
    ├── experiment_framework.py # Common testing framework
    ├── approach_1_test.py      # Test approach 1
    ├── approach_2_test.py      # Test approach 2
    ├── approach_3_test.py      # Test approach 3
    ├── approach_4_test.py      # Test approach 4
    ├── approach_5_test.py      # Test approach 5
    ├── approach_6_test.py      # Test approach 6
    └── compare_all.py          # Compare all approaches
```

---

## 🎯 Success Criteria

### Minimum Viable Success
- ✅ Performance: ≥700 GFLOPS (25% improvement over 566)
- ✅ Correctness: max_error < 0.1
- ✅ Stability: No NaN or Inf values
- ✅ Integration: Works in production engine

### Target Success
- ✅ Performance: ≥900 GFLOPS (60% improvement)
- ✅ Correctness: max_error < 0.01
- ✅ Stability: Robust across matrix sizes
- ✅ Efficiency: Minimal overhead vs. standalone

### Stretch Goal
- ✅ Performance: ≥1100 GFLOPS (near auto-tuner peak)
- ✅ Correctness: max_error < 0.001
- ✅ Stability: Production-grade
- ✅ Generality: Works for multiple tile sizes

---

## 📋 Research Phases

### Phase 1: Setup & Baseline (Week 1)
- [x] Create research directory structure
- [ ] Implement base tile=20 kernel
- [ ] Create experiment framework
- [ ] Establish baseline measurements
- [ ] Document current limitations

### Phase 2: Approach Exploration (Weeks 2-3)
- [ ] Implement Approach 1 (Cooperative)
- [ ] Implement Approach 2 (Non-square)
- [ ] Implement Approach 3 (Transposed)
- [ ] Implement Approach 4 (Hierarchical)
- [ ] Implement Approach 5 (Reduction)
- [ ] Implement Approach 6 (Hybrid)

### Phase 3: Evaluation (Week 4)
- [ ] Benchmark all approaches
- [ ] Compare correctness
- [ ] Analyze performance profiles
- [ ] Identify best candidates

### Phase 4: Optimization (Week 5)
- [ ] Optimize top 2-3 approaches
- [ ] Fine-tune parameters
- [ ] Test edge cases
- [ ] Validate stability

### Phase 5: Integration (Week 6)
- [ ] Integrate best approach into engine
- [ ] Update kernel selection logic
- [ ] Comprehensive testing
- [ ] Documentation

### Phase 6: Decision (Week 7)
- [ ] Compare vs. Phases 2 & 3 alternatives
- [ ] Cost-benefit analysis
- [ ] Final recommendation
- [ ] Merge or archive

---

## 🔄 Workflow

### 1. Experiment Development
```bash
cd research/tile_20_investigation/experiments
python3 approach_X_test.py
```

### 2. Kernel Testing
```bash
cd research/tile_20_investigation
python3 experiments/experiment_framework.py --kernel approach_X
```

### 3. Comparison
```bash
python3 experiments/compare_all.py
```

### 4. Documentation
```bash
# Update EXPERIMENTS_LOG.md with results
# Update FINDINGS.md with insights
```

---

## 🛡️ Safety Guidelines

### DO NOT
- ❌ Modify production kernels in `src/opencl/kernels/`
- ❌ Change production engine in `src/optimization_engines/`
- ❌ Commit experimental code to main branch
- ❌ Break production tests

### DO
- ✅ Work only in `research/tile_20_investigation/`
- ✅ Create new kernel files here
- ✅ Document all experiments
- ✅ Run validation tests frequently
- ✅ Keep production code running

---

## 📊 Expected Outcomes

### Scenario A: Success (≥900 GFLOPS)
→ Integrate into production as FLOAT4_VEC_ULTRA  
→ Update roadmap to prioritize this  
→ Consider further optimizations  

### Scenario B: Partial Success (700-900 GFLOPS)
→ Evaluate vs. Phase 2 & 3 alternatives  
→ Cost-benefit analysis  
→ May integrate or continue research  

### Scenario C: Limited Success (<700 GFLOPS)
→ Document findings  
→ Archive research  
→ Proceed with Phase 2 or 3  

---

## 📚 References

### Auto-Tuner Results
- **Best Config:** T20_L16x16_U4
- **Standalone:** 1148.52 GFLOPS
- **Source:** `scripts/auto_tune_float4_vec.py`
- **Report:** `docs/CONSOLIDATION_REPORT.md`

### Production Kernel
- **File:** `src/opencl/kernels/gemm_float4_clover.cl`
- **Kernel:** `gemm_float4_vec`
- **Performance:** 566 GFLOPS
- **Tile Size:** 16×16

### Related Documentation
- `docs/CONSOLIDATION_REPORT.md`
- `docs/PHASE1_EXTENSION_COMPLETE.md`
- `docs/SESSION29_SUMMARY.md`

---

## 🎓 Learning Objectives

Beyond performance, this research aims to:
1. Understand cooperative memory patterns in GCN architecture
2. Explore tile size vs. thread count trade-offs
3. Investigate synchronization overhead
4. Develop techniques for architectural constraint adaptation
5. Build reusable patterns for future optimizations

---

**Status:** 🚀 READY TO BEGIN  
**Production Code:** ✅ PROTECTED  
**Next Step:** Implement experiment framework
