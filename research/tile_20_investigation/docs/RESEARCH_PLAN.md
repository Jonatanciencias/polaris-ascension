# 🔬 Tile=20 Research Plan

**Created:** Febrero 2026  
**Status:** 🚀 IN PROGRESS  
**Production Code:** ✅ PROTECTED (566 GFLOPS, stable)

---

## 🎯 Research Objective

**Primary Goal:** Achieve **≥900 GFLOPS** (60% improvement over 566 GFLOPS baseline) by integrating tile=20 configuration discovered by auto-tuner.

**Secondary Goal:** Understand architectural constraints and develop techniques for adapting to hardware limitations.

---

## 📊 Baseline & Target

### Current Production (PROTECTED)
```
Kernel:      FLOAT4_VEC
Tile Size:   16×16
Threads:     256 (16×16)
Performance: 566 GFLOPS @ 2048×2048
Correctness: 100% (max_error < 0.001)
Status:      ✅ PRODUCTION READY
```

### Auto-Tuner Discovery
```
Configuration: T20_L16x16_U4
Tile Size:     20×20
Threads:       256 (16×16)
Performance:   1148 GFLOPS (standalone)
Challenge:     256 threads < 400 elements (64% coverage)
```

### Research Targets

| Level | Performance | Improvement | Priority |
|-------|-------------|-------------|----------|
| **Minimum** | 700 GFLOPS | +24% | Must achieve |
| **Target** | 900 GFLOPS | +59% | Goal |
| **Stretch** | 1100 GFLOPS | +94% | Aspirational |

---

## 🔬 Experimental Approaches

### Approach 1: Cooperative Loading ✅ READY TO TEST
**File:** `kernels/approach_1_cooperative.cl`

**Strategy:**
- 256 threads cooperatively load 400 tile elements
- Separate loading phase from compute phase
- Use barriers for synchronization
- Modulo arithmetic for indexing

**Expected Performance:** 700-900 GFLOPS  
**Complexity:** Medium  
**Risk:** Low (well-understood pattern)

**Implementation Status:** ✅ COMPLETE  
**Test Status:** 🔜 PENDING

---

### Approach 2: Non-Square Tiles
**File:** `kernels/approach_2_nonsquare.cl`

**Strategy:**
- Use asymmetric tiles that fit thread geometry better
- Candidates: 16×20, 20×16, 16×24, 24×16
- May reduce wasted threads

**Expected Performance:** 700-850 GFLOPS  
**Complexity:** Medium  
**Risk:** Medium (different memory access patterns)

**Implementation Status:** 📝 TODO  
**Test Status:** ⏸️ PENDING

---

### Approach 3: Transposed Tiles
**File:** `kernels/approach_3_transposed.cl`

**Strategy:**
- Transpose B tile for better memory access
- May reduce LDS bank conflicts
- Cooperative loading for both tiles

**Expected Performance:** 750-950 GFLOPS  
**Complexity:** High  
**Risk:** Medium (complex indexing)

**Implementation Status:** 📝 TODO  
**Test Status:** ⏸️ PENDING

---

### Approach 4: Hierarchical Tiling
**File:** `kernels/approach_4_hierarchical.cl`

**Strategy:**
- Break 20×20 into 4 sub-tiles of 10×10
- Process each sub-tile sequentially
- More synchronization but simpler indexing

**Expected Performance:** 650-800 GFLOPS  
**Complexity:** High  
**Risk:** High (synchronization overhead)

**Implementation Status:** 📝 TODO  
**Test Status:** ⏸️ PENDING

---

### Approach 5: Reduction-Based
**File:** `kernels/approach_5_reduction.cl`

**Strategy:**
- Multiple threads compute partial sums
- Reduce within work-group
- Natural fit for thread count

**Expected Performance:** 600-750 GFLOPS  
**Complexity:** Very High  
**Risk:** High (reduction overhead)

**Implementation Status:** 📝 TODO  
**Test Status:** ⏸️ PENDING

---

### Approach 6: Hybrid Dynamic Selection
**File:** `kernels/approach_6_hybrid.cl`

**Strategy:**
- Use tile=20 for large matrices (≥2048)
- Use tile=16 for medium matrices
- Runtime selection based on size

**Expected Performance:** Best of both worlds  
**Complexity:** Low (integration)  
**Risk:** Low (fallback available)

**Implementation Status:** 📝 TODO  
**Test Status:** ⏸️ PENDING

---

## 📅 Timeline

### Week 1: Approach 1 (Current)
- [x] Setup research directory
- [x] Create experiment framework
- [x] Implement Approach 1 kernel
- [x] Create test script
- [ ] **Run tests and analyze results**
- [ ] Document findings

### Week 2: Approaches 2-3
- [ ] Implement Approach 2 (Non-square)
- [ ] Test Approach 2
- [ ] Implement Approach 3 (Transposed)
- [ ] Test Approach 3
- [ ] Comparative analysis

### Week 3: Approaches 4-5
- [ ] Implement Approach 4 (Hierarchical)
- [ ] Test Approach 4
- [ ] Implement Approach 5 (Reduction)
- [ ] Test Approach 5
- [ ] Identify top 2 candidates

### Week 4: Optimization
- [ ] Optimize top 2 approaches
- [ ] Parameter tuning
- [ ] Edge case testing
- [ ] Stability validation

### Week 5: Integration & Decision
- [ ] Integrate best approach into engine
- [ ] Production testing
- [ ] Compare vs Phase 2 & 3 alternatives
- [ ] Final recommendation

---

## 📊 Success Criteria

### Technical Criteria

✅ **Performance:**
- Minimum: ≥700 GFLOPS (24% improvement)
- Target: ≥900 GFLOPS (59% improvement)
- Stretch: ≥1100 GFLOPS (94% improvement)

✅ **Correctness:**
- Maximum error: <0.1 for acceptance
- Preferred: <0.01 for production
- No NaN or Inf values

✅ **Stability:**
- Works across matrix sizes (512-4096)
- Consistent performance
- No regressions vs. baseline

✅ **Integration:**
- Fits within current engine architecture
- No breaking changes to API
- Backward compatible

### Strategic Criteria

📊 **Cost-Benefit:**
- Performance gain > complexity cost
- Maintenance burden acceptable
- Documentation complete

🔄 **Comparison:**
- Better than or competitive with Phase 2 alternatives
- Justifies research investment
- Clear path to production

---

## 🛡️ Risk Management

### Risks

1. **Performance below target (<700 GFLOPS)**
   - Mitigation: Multiple approaches, early testing
   - Fallback: Archive research, proceed to Phase 2

2. **Correctness issues persist**
   - Mitigation: Comprehensive validation framework
   - Fallback: Conservative tolerance, extensive testing

3. **Integration complexity too high**
   - Mitigation: Modular design, clear interfaces
   - Fallback: Standalone kernel, manual selection

4. **Maintenance burden**
   - Mitigation: Extensive documentation
   - Fallback: Simplify or archive

### Safety Net

✅ **Production code is PROTECTED**
- All experiments in separate directory
- No changes to `src/` during research
- Can revert completely at any time

✅ **Validation framework**
- Automated correctness checking
- Performance regression detection
- Consistent benchmarking

---

## 📝 Documentation Standards

### Required for Each Approach

1. **Kernel Code:**
   - Inline comments explaining strategy
   - Author and date
   - Status marker (EXPERIMENTAL)

2. **Test Results:**
   - Performance across all test sizes
   - Correctness metrics
   - Status (SUCCESS/FAIL)

3. **Analysis:**
   - What worked
   - What didn't work
   - Key insights

4. **Findings Document:**
   - Technical lessons
   - Architectural insights
   - Recommendations

---

## 🎓 Learning Objectives

Beyond performance, this research aims to:

1. **Understand GCN cooperative patterns**
   - How to efficiently use 256 threads for 400 elements
   - Synchronization overhead analysis
   - Optimal loading strategies

2. **Explore tile size trade-offs**
   - Performance vs. complexity
   - Memory bandwidth vs. compute
   - Synchronization costs

3. **Develop adaptation techniques**
   - Working within hardware constraints
   - Creative solutions to architectural limits
   - Reusable patterns for future work

4. **Build research methodology**
   - Systematic experimentation
   - Rigorous validation
   - Clear documentation

---

## 📈 Progress Tracking

### Current Status (Week 1, Day 1)

**Completed:**
- ✅ Research directory structure
- ✅ Experiment framework (470 lines)
- ✅ Approach 1 kernel (153 lines)
- ✅ Approach 1 test script (100 lines)
- ✅ Documentation (README, this plan)

**In Progress:**
- 🔄 Testing Approach 1

**Pending:**
- ⏸️ Approaches 2-6 implementation
- ⏸️ Comparative analysis
- ⏸️ Optimization phase
- ⏸️ Integration decision

---

## 🔄 Next Steps

### Immediate (Today)
1. ✅ Complete setup
2. 🔜 **Run Approach 1 tests**
3. 📊 Analyze results
4. 📝 Document findings

### Short-term (This Week)
1. Iterate on Approach 1 if needed
2. Begin Approach 2 implementation
3. Update progress tracking

### Medium-term (Weeks 2-4)
1. Complete all 6 approaches
2. Identify best candidates
3. Optimization phase

---

**Status:** 🚀 ACTIVE RESEARCH  
**Next Milestone:** Approach 1 test results  
**Decision Point:** Week 5 (integration vs. archive)
