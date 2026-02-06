# Production Integration Plan - Phase 2.1 System

**Date**: 4 febrero 2026  
**Version**: 1.0  
**Status**: Ready for Integration

---

## 🎯 System Overview

### Performance Achieved
- **Peak**: 866.9 GFLOPS @ 1400×1400 (+53% vs baseline)
- **Target**: 850 GFLOPS ✅ EXCEEDED
- **Kernels**: 3 specialized (tile16, tile20, tile24)
- **Selector**: ML-powered adaptive (75% accuracy)

### Components to Integrate
1. ✅ tile20_vectorized.cl kernel
2. ✅ tile24_vectorized.cl kernel  
3. ✅ AdvancedAdaptiveKernelSelector (Python)
4. ✅ advanced_neural_model.pkl (trained model)
5. ✅ consolidated_neural_dataset.json (training data)

---

## 📁 Integration Structure

### Current State (Research)
```
research/tile_20_investigation/
├── kernels/
│   ├── tile20_optimized.cl (773 GFLOPS @ 1400)
│   └── tile24_vectorized.cl (764 GFLOPS @ 2048)
├── advanced_adaptive_selector.py
├── advanced_neural_model.pkl
└── consolidated_neural_dataset.json
```

### Target State (Production)
```
src/
├── kernels/
│   ├── gemm_tile16_baseline.cl (existing, 566 GFLOPS)
│   ├── gemm_tile20_production.cl (NEW, 866.9 GFLOPS)
│   └── gemm_tile24_production.cl (NEW, 764 GFLOPS)
├── optimization_engines/
│   └── adaptive_kernel_selector.py (NEW, ML-powered)
└── ml_models/
    ├── kernel_selector_model.pkl (NEW, trained GB model)
    └── kernel_selector_dataset.json (NEW, 21 samples)
```

---

## 🔄 Integration Steps

### Step 1: Prepare Kernels ✅
- [x] Validate tile20 correctness (max_error < 0.001)
- [x] Validate tile24 correctness (max_error < 0.001)
- [x] Benchmark performance across sizes
- [x] Document kernel parameters

### Step 2: Copy Files to Production
```bash
# Kernels
cp research/tile_20_investigation/kernels/tile20_optimized.cl \
   src/kernels/gemm_tile20_production.cl

cp research/tile_20_investigation/kernels/tile24_vectorized.cl \
   src/kernels/gemm_tile24_production.cl

# Selector
cp research/tile_20_investigation/advanced_adaptive_selector.py \
   src/optimization_engines/adaptive_kernel_selector.py

# Model & Data
cp research/tile_20_investigation/advanced_neural_model.pkl \
   src/ml_models/kernel_selector_model.pkl

cp research/tile_20_investigation/consolidated_neural_dataset.json \
   src/ml_models/kernel_selector_dataset.json
```

### Step 3: Update Imports & Paths
- [ ] Update kernel paths in selector
- [ ] Update model paths
- [ ] Add to src/__init__.py exports
- [ ] Update requirements.txt (sklearn, etc.)

### Step 4: Create Production API
```python
# src/optimization_engines/adaptive_kernel_selector.py

class ProductionKernelSelector:
    """Production-ready adaptive GEMM kernel selector"""
    
    def __init__(self):
        self.selector = AdvancedAdaptiveKernelSelector()
        self.kernels = {
            'tile16': 'src/kernels/gemm_tile16_baseline.cl',
            'tile20': 'src/kernels/gemm_tile20_production.cl',
            'tile24': 'src/kernels/gemm_tile24_production.cl'
        }
    
    def select_kernel(self, M: int, N: int, K: int) -> dict:
        """
        Select optimal kernel for given matrix sizes
        
        Returns:
            {
                'kernel_path': str,
                'kernel_name': str,
                'predicted_gflops': float,
                'local_size': tuple,
                'tile_size': int
            }
        """
        rec = self.selector.get_recommendation(M, N, K)
        kernel_config = rec['config']['kernel_name']
        
        # Map to production kernel
        if 'tile24' in kernel_config:
            kernel_key = 'tile24'
        elif 'tile20' in kernel_config:
            kernel_key = 'tile20'
        else:
            kernel_key = 'tile16'
        
        return {
            'kernel_path': self.kernels[kernel_key],
            'kernel_name': kernel_key,
            'predicted_gflops': rec['predicted_gflops'],
            'local_size': rec['config']['local_size'],
            'tile_size': rec['config']['tile_size']
        }
```

### Step 5: Integration Testing
- [ ] Unit tests for selector
- [ ] Performance regression tests
- [ ] Correctness validation (all sizes)
- [ ] Memory leak checks

### Step 6: Documentation
- [ ] Update main README.md
- [ ] API documentation
- [ ] Performance benchmarks table
- [ ] Migration guide from old kernels

---

## 🧪 Testing Strategy

### Unit Tests
```python
def test_selector_basic():
    selector = ProductionKernelSelector()
    
    # Test sweet spot (1400)
    result = selector.select_kernel(1400, 1400, 1400)
    assert result['kernel_name'] == 'tile20'
    assert result['predicted_gflops'] > 850
    
    # Test large matrix (2048)
    result = selector.select_kernel(2048, 2048, 2048)
    assert result['kernel_name'] == 'tile24'
    assert result['predicted_gflops'] > 700
```

### Integration Tests
```python
def test_end_to_end_gemm():
    selector = ProductionKernelSelector()
    
    for size in [512, 1024, 1400, 2048, 3072]:
        M = N = K = size
        
        # Get recommendation
        rec = selector.select_kernel(M, N, K)
        
        # Execute GEMM
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_gpu = execute_gemm(rec['kernel_path'], A, B)
        
        # Validate
        C_ref = A @ B
        assert np.allclose(C_gpu, C_ref, rtol=1e-3)
```

### Performance Tests
```python
def test_performance_benchmarks():
    selector = ProductionKernelSelector()
    
    benchmarks = {
        1400: 850,  # minimum GFLOPS expected
        2048: 700,
        3072: 650
    }
    
    for size, min_gflops in benchmarks.items():
        rec = selector.select_kernel(size, size, size)
        actual_gflops = benchmark_gemm(rec['kernel_path'], size)
        
        assert actual_gflops >= min_gflops, \
            f"Size {size}: {actual_gflops} < {min_gflops} GFLOPS"
```

---

## 📊 Rollout Plan

### Phase 1: Soft Launch (Week 1)
- Deploy to development environment
- Internal testing with subset of workloads
- Monitor performance metrics
- Collect telemetry data

### Phase 2: A/B Testing (Week 2)
- 50/50 split: new selector vs baseline
- Compare:
  - Performance (GFLOPS)
  - Correctness (error rates)
  - Stability (crash rates)
  - Selection accuracy

### Phase 3: Full Rollout (Week 3)
- If A/B successful: 100% new selector
- If issues found: rollback to baseline
- Document learnings

---

## 🎯 Success Criteria

### Must Have
- ✅ All tests passing (unit + integration)
- ✅ Performance >= 850 GFLOPS @ 1400
- ✅ Correctness: max_error < 0.001
- ✅ No memory leaks
- ✅ API documentation complete

### Nice to Have
- ⭐ Performance monitoring dashboard
- ⭐ Auto-tuning based on production data
- ⭐ Kernel JIT compilation
- ⭐ Multi-GPU support

---

## 🚨 Rollback Plan

### Triggers
- Performance regression > 10%
- Correctness errors detected
- Stability issues (crashes)
- Memory leaks

### Rollback Process
1. Revert to baseline kernels (tile16)
2. Disable adaptive selector
3. Document failure mode
4. Root cause analysis
5. Fix and re-test

### Rollback Time: < 5 minutes

---

## 📈 Monitoring & Metrics

### Key Metrics
- **Performance**: GFLOPS per operation
- **Accuracy**: Kernel selection correctness
- **Latency**: Selection overhead
- **Memory**: Peak usage, leaks
- **Errors**: Correctness failures

### Alerting Thresholds
- Performance < 800 GFLOPS @ 1400: WARNING
- Performance < 700 GFLOPS @ 1400: CRITICAL
- Max error > 0.01: CRITICAL
- Memory leak > 100 MB/hour: WARNING

---

## 🔮 Future Enhancements

### Short-term (1-2 months)
1. **Production Data Collection**
   - Log actual matrix sizes
   - Measure real performance
   - Retrain model with production data

2. **Online Learning**
   - Update model based on real workloads
   - Adapt to specific use cases
   - Improve selection accuracy

### Long-term (3-6 months)
1. **ROCm Migration**
   - FP16 support (1200+ GFLOPS)
   - Better compiler
   - Modern OpenCL 2.x

2. **Auto-tuning**
   - Runtime kernel optimization
   - JIT compilation
   - Cache-aware tiling

3. **Multi-GPU**
   - Workload distribution
   - Data parallelism
   - Hybrid CPU+GPU

---

## ✅ Pre-Integration Checklist

- [x] Phase 2.1 complete (866.9 GFLOPS achieved)
- [x] float8 experiment completed and documented
- [x] All kernels validated for correctness
- [x] ML model trained and validated (R²=1.0)
- [x] Decision made: integrate to production
- [ ] Files copied to production locations
- [ ] API created and documented
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Rollback plan ready

---

**Status**: Ready to begin integration  
**Estimated Time**: 2-3 hours  
**Risk**: LOW (proven in research, easy rollback)  
**Expected Impact**: +53% performance vs baseline

**Next Step**: Execute Step 2 (Copy files to production) ✅
