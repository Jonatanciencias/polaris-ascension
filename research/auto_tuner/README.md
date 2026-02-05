# GEMM Auto-Tuner Framework

**Status**: ✅ Running (in progress)  
**Date**: February 5, 2026  
**Expected completion**: 30-60 minutes

## Overview

Automated parameter search to find optimal GEMM kernel configuration for AMD RX 590.

### Search Space

**Kernels tested**:
- `tile20` (10×10 workgroup, 20-element tiles)
- `tile24` (12×12 workgroup, 24-element tiles)

**Matrix sizes** (21 sizes):
- Sweet spot region: 1200, 1250, 1300, 1350, 1375, 1400, 1425, 1450, 1500, 1550, 1600
- Medium: 1700, 1800, 1900, 2000
- Large: 2048, 2560, 3072, 3584, 4096
- Extra large: 5120

**Benchmark protocol**:
- 10 runs per configuration
- 2 warmup runs
- Profiling enabled
- Correctness verification (max_error < 0.01)

**Total configurations**: 42 (2 kernels × 21 matrix sizes)

---

## Current Status

**Running**: `python3 research/auto_tuner/gemm_auto_tuner.py`

**Progress**: ~47.6% (20/42 configurations tested)

**Current test**: tile20 @ 4096×4096

---

## Output Files

Results are saved incrementally to `results/auto_tuner/`:

### 1. `tuning_results.csv`
Complete results for all tested configurations:
```csv
tile_size,matrix_size,workgroup_x,workgroup_y,gflops,avg_time_ms,max_error,timestamp,runs
20,1200,10,10,805.2,7.0,0.000234,2026-02-05 14:30:15,10
...
```

### 2. `tuning_summary.json`
Summary statistics and top configurations:
```json
{
  "elapsed_time_minutes": 45.2,
  "total_configurations": 42,
  "best_configuration": {
    "tile_size": 20,
    "matrix_size": 1400,
    "gflops": 810.5,
    ...
  },
  "top_10_configurations": [...],
  "kernel_summary": {
    "tile20": {"max_gflops": 810.5, "mean_gflops": 765.3, ...},
    "tile24": {"max_gflops": 710.2, "mean_gflops": 680.1, ...}
  }
}
```

---

## Expected Outcomes

### Scenario A: Confirmation (80% probability)
- **Result**: tile20 @ 1400 remains optimal
- **Performance**: ~805-810 GFLOPS (matches current best)
- **Conclusion**: Current configuration is validates as optimal

### Scenario B: Minor Discovery (15% probability)
- **Result**: tile20 @ [1350, 1450, or 1500] performs slightly better
- **Performance**: ~812-820 GFLOPS (+0.5-1.5%)
- **Conclusion**: Small sweet spot adjustment

### Scenario C: Surprise Discovery (5% probability)
- **Result**: tile24 @ specific size outperforms tile20
- **Performance**: ~820-830 GFLOPS (+2-3%)
- **Conclusion**: Unexpected configuration found

---

## Usage

### Run auto-tuner:
```bash
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580
python3 research/auto_tuner/gemm_auto_tuner.py
```

### Monitor progress:
```bash
# Check terminal output (shows live progress)
# Or check results file
wc -l results/auto_tuner/tuning_results.csv
```

### Analyze results:
```bash
# View top configurations
cat results/auto_tuner/tuning_summary.json | jq '.top_10_configurations'

# Plot results
python research/auto_tuner/plot_results.py
```

---

## Implementation Details

### Kernel loading
- Dynamically loads OpenCL kernels from `src/kernels/`
- Verifies kernel existence before benchmarking
- Handles compilation errors gracefully

### Benchmarking protocol
1. Allocate random test matrices (CPU)
2. Transfer to GPU buffers
3. Warmup runs (2×) - prime cache, stabilize clocks
4. Timed runs (10×) - profiling enabled
5. Verify correctness - compare vs CPU GEMM
6. Calculate statistics - mean, min GFLOPS
7. Track best configuration

### Error handling
- Correctness check: max_error < 0.01
- Failed configurations skipped
- Partial results saved (Ctrl+C safe)
- Detailed error logging

---

## Performance Analysis

### Key metrics tracked
- **GFLOPS**: Throughput (higher = better)
- **Time (ms)**: Latency per run
- **Error**: Numerical accuracy
- **Stability**: Standard deviation across runs

### Per-kernel statistics
- Max, mean, min GFLOPS
- Best matrix size  
- Configuration count

---

## Next Steps

After tuning completes:

1. **Review results**: Compare vs current best (810 GFLOPS)
2. **Update docs**: If new optimal found, update README.md
3. **Visualize**: Create performance heatmaps (tile×size)
4. **Share findings**: Document in auto-tuner report
5. **Proceed to publication**: If confirmed, ready to publish

---

## Technical Notes

### Search strategy
- **Grid search**: Exhaustive (not random/Bayesian)
- **Rationale**: Small search space (42 configs), want completeness
- **Runtime**: ~30-60 min (acceptable for comprehensive results)

### Why these sizes?
- **1200-1600**: Sweet spot region (fine-grained)
- **1400**: Current best (need to validate)
- **2048-5120**: Large matrix validation
- **Powers of 2**: Common ML workload sizes

### Why tile20 and tile24?
- **tile20**: Current champion (810 GFLOPS @ 1400)
- **tile24**: Large matrix specialist (710 GFLOPS @ 3072)
- **Excluded**: tile16 (slower baseline), tile32 (negative EV)

---

## Author

GEMM Optimization Project  
AMD RX 590 GEMM Auto-Tuner  
February 5, 2026
