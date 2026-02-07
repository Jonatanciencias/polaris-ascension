# 📚 Documentation Index

**AMD RX 590 GEMM Optimization Framework**  
Complete documentation structure and navigation guide.

---

## 🎯 Start Here

New to this project? Start with these documents in order:

1. [../README.md](../README.md) - Project overview, quick start, performance results
2. [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) - Complete assessment, novelty analysis, publication recommendations
3. [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) - Verified performance on real RX 590 hardware
4. [PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md](PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md) - Reproducible baseline protocol (Feb 7, 2026)

---

## 📖 Main Documentation

### Project Overview
- [README.md](../README.md) - Main project documentation
  - Quick start guide
  - Performance benchmarks
  - Architecture overview
  - Installation instructions

### Assessment & Validation
- [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) ⭐
  - Complete project evaluation
  - Novelty assessment (⭐⭐⭐⭐ engineering, ⭐⭐ research)
  - Publication recommendations (Blog+GitHub: yes, Workshop: maybe, Conference: no)
  - Honest competitive analysis
  - Real results: 750-805 GFLOPS (+37-42% improvement)

- [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) ⭐
  - Verified benchmarks on AMD RX 590 GME
  - Conservative performance claims (750-805 GFLOPS)
  - Comparison: claimed vs actual results
  - Correctness validation (max_error < 0.001)

- [PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md](PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md) ⭐
  - Fixed protocol: 10 sessions x 20 iterations, seed=42
  - Reproducible peak baseline: 776.1 GFLOPS @ 1400x1400
  - Historical discovery separated from reproducible claims

- [test_production_system.py](../test_production_system.py) ⭐
  - Comprehensive validation suite (4 tests)
  - Production selector test
  - File integrity verification
  - Real hardware performance validation
  - Novelty analysis output

---

## 🔬 Research Journey

### Phase 2.1: Sweet Spot + Tile24
- [research/tile_20_investigation/PHASE21_FINAL_REPORT.md](../research/tile_20_investigation/PHASE21_FINAL_REPORT.md)
  - tile20 sweet spot discovery (778 GFLOPS @ 1400×1400)
  - tile24 scaling investigation (805 GFLOPS @ 3072×3072)
  - ML selector training (R²=1.0, 75% cross-val accuracy)
  - Complete methodology

### Failed Experiments
- [research/tile_20_investigation/FLOAT8_EXPERIMENT.md](../research/tile_20_investigation/FLOAT8_EXPERIMENT.md)
  - float8 vectorization attempt
  - Result: -60% performance (773→307 GFLOPS)
  - Root cause: Register spilling (preferred width=4, not 8)
  - Learning: Hardware constraints matter more than theory
  - Time: 2.5 hours (acceptable risk)

### Production Integration
- [research/tile_20_investigation/INTEGRATION_COMPLETE.md](../research/tile_20_investigation/INTEGRATION_COMPLETE.md)
  - Production deployment summary
  - Files copied to src/
  - Integration validation
  - Production-ready checklist

### Earlier Phases
- [research/tile_20_investigation/PHASE22_FP16_REPORT.md](../research/tile_20_investigation/PHASE22_FP16_REPORT.md)
  - FP16 investigation (blocked by driver)
- [research/tile_20_investigation/PRODUCTION_INTEGRATION_PLAN.md](../research/tile_20_investigation/PRODUCTION_INTEGRATION_PLAN.md)
  - Original integration roadmap

---

## 🏗️ Architecture & Technical Details

### System Design
- [architecture.md](architecture.md)
  - System architecture overview
  - Component interaction
  - Data flow diagrams

### Optimization Techniques
- [optimization.md](optimization.md)
  - Tile-size optimization methodology
  - float4 vectorization
  - Register blocking strategies
  - Loop unrolling techniques

### Kernel Compilation
- [KERNEL_CACHE.md](KERNEL_CACHE.md)
  - Kernel compilation caching
  - Performance improvements
  - Cache management

---

## 📊 Benchmarks & Performance

### Verified Performance Data
See [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) for complete benchmark results.

**Quick Reference**:
- Reproducible peak baseline: 776.1 GFLOPS @ 1400x1400 (tile20)
- Reproducible large-matrix baseline: 774.3 GFLOPS @ 2048x2048 (tile24)
- Baseline: 566 GFLOPS (tile16 @ 2048x2048)
- Reproducible improvement: ~+37.1%
- Historical discovery: 831.2 GFLOPS @ 1300x1300 (auto-tuner archive)
- Correctness: max_error < 0.001 on all sizes

### Hardware Used
- **GPU**: AMD Radeon RX 590 GME
- **Driver**: Mesa Clover (OpenCL 1.1)
- **Platform**: radeonsi, Polaris10, ACO compiler
- **OS**: Ubuntu Linux 6.14.0-37-generic

---

## 🧪 Testing & Validation

### Production Test Suite
- [test_production_system.py](../test_production_system.py)
  - Test 1: Production selector functionality
  - Test 2: File integrity checks
  - Test 3: Real hardware benchmarks
  - Test 4: Novelty assessment

### Running Tests
```bash
# Complete validation
python test_production_system.py

# Expected: 4/4 tests pass ✅
```

---

## 🔧 Development & Contributing

### Contributing Guidelines
- [CONTRIBUTING.md](../CONTRIBUTING.md)
  - How to contribute
  - Code standards
  - Testing requirements

### Project Metadata
- [LICENSE](../LICENSE) - MIT License
- [setup.py](../setup.py) - Package configuration
- [requirements.txt](../requirements.txt) - Dependencies
- [pyproject.toml](../pyproject.toml) - Build configuration

---

## 📝 Historical Documentation

### Progress Tracking (Archive)
- [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md)
- [ROADMAP_CHECKLIST.md](ROADMAP_CHECKLIST.md)
- [ROADMAP_OPTIMIZATION.md](ROADMAP_OPTIMIZATION.md)
- [SYSTEM_STATUS_REPORT.md](SYSTEM_STATUS_REPORT.md)

### Session Reports (Archive)
- [SESSION29_SUMMARY.md](SESSION29_SUMMARY.md)
- [VALIDATION_REPORT_SESSION29.md](VALIDATION_REPORT_SESSION29.md)
- [PHASE1_COMPLETION_REPORT.md](PHASE1_COMPLETION_REPORT.md)
- [PHASE1_EXTENSION_COMPLETE.md](PHASE1_EXTENSION_COMPLETE.md)
- [PHASE1_INTEGRATION_REPORT.md](PHASE1_INTEGRATION_REPORT.md)

### Consolidation Reports (Archive)
- [CONSOLIDACION_RESUMEN_ESPAÑOL.md](CONSOLIDACION_RESUMEN_ESPAÑOL.md)
- [CONSOLIDATION_EXECUTIVE_SUMMARY.md](CONSOLIDATION_EXECUTIVE_SUMMARY.md)
- [CONSOLIDATION_REPORT.md](CONSOLIDATION_REPORT.md)

**Note**: Historical documents reflect earlier project phases when the focus was broader (energy efficiency, multiple algorithms). Current production system focuses specifically on GEMM optimization with 3 specialized kernels.

---

## 🗂️ Documentation by Topic

### For Users
1. [README.md](../README.md) - Getting started
2. [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) - Performance expectations
3. [test_production_system.py](../test_production_system.py) - Validation

### For Developers
1. [architecture.md](architecture.md) - System design
2. [optimization.md](optimization.md) - Optimization techniques
3. [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
4. [research/tile_20_investigation/](../research/tile_20_investigation/) - Research process

### For Researchers
1. [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) - Complete assessment
2. [research/tile_20_investigation/PHASE21_FINAL_REPORT.md](../research/tile_20_investigation/PHASE21_FINAL_REPORT.md) - Methodology
3. [research/tile_20_investigation/FLOAT8_EXPERIMENT.md](../research/tile_20_investigation/FLOAT8_EXPERIMENT.md) - Failed experiments
4. [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) - Verified results

### For Publishers/Bloggers
1. [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) - Publication recommendations
2. [REAL_HARDWARE_VALIDATION.md](../REAL_HARDWARE_VALIDATION.md) - Honest results
3. [research/tile_20_investigation/PHASE21_FINAL_REPORT.md](../research/tile_20_investigation/PHASE21_FINAL_REPORT.md) - Complete methodology
4. [research/tile_20_investigation/FLOAT8_EXPERIMENT.md](../research/tile_20_investigation/FLOAT8_EXPERIMENT.md) - Lessons learned

---

## 🔍 Quick Lookup

### Performance Numbers
- **Reproducible peak GFLOPS**: 776.1 ([PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md](PHASE3_REPRODUCIBLE_PERFORMANCE_BASELINE_FEB2026.md))
- **Reproducible improvement**: ~+37.1% ([../README.md](../README.md))
- **Historical discovery peak**: 831.2 GFLOPS ([../AUTO_TUNER_COMPLETE_SUMMARY.md](../AUTO_TUNER_COMPLETE_SUMMARY.md))

### Key Files
- **Production Selector**: `src/optimization_engines/adaptive_kernel_selector.py`
- **tile20 Kernel**: `src/kernels/gemm_tile20_production.cl`
- **tile24 Kernel**: `src/kernels/gemm_tile24_production.cl`
- **ML Model**: `src/ml_models/kernel_selector_model.pkl`

### Research Phases
- **Phase 2.1**: Sweet spot + tile24 ([PHASE21_FINAL_REPORT.md](../research/tile_20_investigation/PHASE21_FINAL_REPORT.md))
- **float8 Experiment**: Failed ([FLOAT8_EXPERIMENT.md](../research/tile_20_investigation/FLOAT8_EXPERIMENT.md))
- **Production Integration**: Complete ([INTEGRATION_COMPLETE.md](../research/tile_20_investigation/INTEGRATION_COMPLETE.md))

### Publication Guidance
- **Blog Post**: ✅ Recommended ([EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md))
- **GitHub**: ✅ Recommended
- **Workshop**: ⚠️ Maybe (if extended)
- **Conference**: ❌ Limited research novelty

---

## 📞 Support & Contact

- **Issues**: Open GitHub issue
- **Questions**: See [EXECUTIVE_SUMMARY.md](../EXECUTIVE_SUMMARY.md) publication section
- **Contributions**: Read [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Last Updated**: February 2026  
**Status**: Production Ready ✅  
**Documentation Coverage**: Complete 🎯
