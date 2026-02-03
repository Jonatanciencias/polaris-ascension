# Task 1.1.3: Memory Optimization - Plan Detallado

**Status:** 🟡 EN PROGRESO  
**Fecha:** 2026-01-24  
**Duración Estimada:** 4 horas  
**Prioridad:** CRÍTICA (Cierre de Phase 1)  

---

## 📋 Resumen Ejecutivo

**Objetivo:** Optimizar patrones de acceso a memoria del kernel híbrido para alcanzar 750-800 GFLOPS.

**Entrada:** Task 1.1.2 baseline (650-700 GFLOPS esperado)  
**Salida:** Kernel optimizado (750-800 GFLOPS)  
**Ganancia Esperada:** +15-20%  

---

## 🎯 Objetivos de Task 1.1.3

### Objetivo Primario
> Optimizar acceso a memoria para eliminar cuellos de botella identificados en Task 1.1.2.

### Sub-objetivos

| # | Objetivo | Meta | Duración |
|---|----------|------|----------|
| 1.1.3.1 | LDS bank conflict tuning | Reducir conflictos | 1.5h |
| 1.1.3.2 | Memory coalescing optimization | Maximizar throughput | 1.0h |
| 1.1.3.3 | Register allocation refinement | Reducir spills | 1.0h |
| 1.1.3.4 | Full validation & reporting | Performance gain ≥15% | 0.5h |

---

## 🔍 Análisis de Oportunidades

### Del Task 1.1.2 Analysis

**LDS Bank Conflicts:** Detectados
- Actual: 4-byte padding per row (16×17 structure)
- Problema: Access patterns pueden causar conflicts
- Solución: Fine-tune padding size
- Ganancia estimada: +3-5%

**Memory Coalescing:** Suboptimal
- Actual: float4 loads coalesced
- Problema: Stores a LDS no optimizados
- Solución: Verify coalescing patterns en stores
- Ganancia estimada: +5-8%

**Register Allocation:** Moderado
- Actual: ~24 registers/thread
- Problema: Temporaries innecesarios
- Solución: Reduce temporaries, inline more
- Ganancia estimada: +3-5%

### Total Esperado: +15-20% (75-150 GFLOPS)

---

## 🛠️ Desglose de Subtasks

### Subtask 1.1.3.1: LDS Bank Conflict Optimization (1.5h)

**Objetivo:** Minimizar conflictos de acceso a LDS

**Análisis:**
```
Hardware: GCN 4.0 (Polaris 10)
- 32 banks de 4 bytes cada
- Accesos simultáneos a mismo banco = conflicto

Patrón actual:
  float A_tile[TILE_SIZE][TILE_SIZE + PADDING];
  TILE_SIZE = 16, PADDING = 4
  Structure: 16×20 floats = 16×80 bytes

Potential issue:
  Si PADDING=4 bytes (1 float), puede no ser suficiente
  para evitar todos los conflictos
```

**Solución:**
1. Aumentar PADDING a 8 bytes (2 floats)
2. Analizar access patterns en detalle
3. Medir impact en performance
4. Seleccionar óptimo

**Archivos:**
- Kernel optimizado: `src/opencl/kernels/gemm_hybrid_opt.cl`
- Script de análisis: `scripts/analyze_bank_conflicts.py`
- Script de comparación: `scripts/compare_lds_variants.py`

**Métricas:**
- Bank conflict ratio
- Effective bandwidth
- Performance delta

### Subtask 1.1.3.2: Memory Coalescing Optimization (1.0h)

**Objetivo:** Maximizar memory bandwidth utilization

**Análisis:**
```
Loads (global memory):
  - Coalesced bien: float4 vectorization
  - Status: ✅ BUENO

Stores (a LDS):
  - Patrón: Scatter writes a LDS
  - Issue: Puede no ser óptimo
  - Target: Verify y optimizar si necesario

Output writes (global C):
  - Status: Check coalescing patterns
  - Opportunity: Verify stride patterns
```

**Soluciones Posibles:**
1. Verify coalescing en stores a LDS
2. Check global output write patterns
3. Optimize write-back order
4. Consider memory layout changes

**Archivos:**
- Script de análisis: `scripts/analyze_coalescing.py`
- Kernel optimizado: Updates en `gemm_hybrid_opt.cl`

**Métricas:**
- Transaction efficiency
- Bandwidth utilization %
- Write patterns analysis

### Subtask 1.1.3.3: Register Allocation Refinement (1.0h)

**Objetivo:** Optimizar uso de registros

**Análisis:**
```
Current register usage: ~24/thread
- Accumulators: 4 (2×2 blocking)
- Temp variables: 8
- LDS pointers: 2-4
- Loop counters: 3-4
- Others: 6-8

Optimization opportunities:
- Eliminate unnecessary temporaries
- Inline calculations
- Use local variables efficiently
- Reduce register pressure
```

**Soluciones:**
1. Analyze register allocation report
2. Refactor hot paths
3. Inline simple calculations
4. Reduce variable scope
5. Benchmark impact

**Archivos:**
- Kernel optimizado: Updates en `gemm_hybrid_opt.cl`
- Script de análisis: `scripts/analyze_register_usage.py`

**Métricas:**
- Register usage/thread
- Spill rate
- Occupancy impact

### Subtask 1.1.3.4: Full Validation (0.5h)

**Objetivo:** Validar todas las optimizaciones

**Tests:**
1. ✅ Correctness: Error < 1e-4
2. ✅ Performance: Baseline + delta
3. ✅ Stability: CV < 1%
4. ✅ Regression: No performance loss

**Archivos:**
- Script de validación: `scripts/validate_optimizations.py`
- Script de comparación: `scripts/compare_kernels.py`
- Reporte final: `TASK_1_1_3_FINAL_REPORT.md`

---

## 📊 Performance Targets

### Baseline (Task 1.1.2)
```
Expected: 650-700 GFLOPS
Conservative: 650 GFLOPS
Optimistic: 700 GFLOPS
```

### After Task 1.1.3 Optimizations
```
LDS opt (+3-5%):        670-735 GFLOPS
Coalescing (+5-8%):     700-795 GFLOPS
Register opt (+3-5%):   720-835 GFLOPS
─────────────────────────────────────
PHASE 1 TARGET:         750-800 GFLOPS ✅
```

---

## 🔧 Archivos a Crear/Modificar

### Nuevos Kernels

```
src/opencl/kernels/
├── gemm_hybrid.cl (original)
└── gemm_hybrid_opt.cl (NEW - optimized variants)
    ├── gemm_hybrid_lds_opt (LDS padding tuning)
    ├── gemm_hybrid_coalesce_opt (Memory coalescing)
    └── gemm_hybrid_full_opt (All optimizations)
```

### Scripts de Optimización

```
scripts/
├── analyze_bank_conflicts.py (NEW)
├── analyze_coalescing.py (NEW)
├── analyze_register_usage.py (NEW)
├── compare_lds_variants.py (NEW)
├── compare_kernels.py (NEW)
└── validate_optimizations.py (NEW)
```

### Documentación

```
docs/
└── MEMORY_OPTIMIZATION_GUIDE.md (NEW)

Root/
├── TASK_1_1_3_PLAN.md (THIS FILE)
├── TASK_1_1_3_STATUS.md (NEW)
└── TASK_1_1_3_FINAL_REPORT.md (NEW)
```

---

## ✅ Criterios de Aceptación

### Compilación
- [ ] Kernel compila sin errores
- [ ] Warnings < 5
- [ ] Compilación < 10 segundos

### Funcionalidad
- [ ] Correctness test: PASS (error <1e-4)
- [ ] All alpha/beta combinations: PASS
- [ ] Stability: CV < 1%

### Performance
- [ ] Baseline: ≥650 GFLOPS
- [ ] After LDS opt: ≥670 GFLOPS
- [ ] After coalesce opt: ≥700 GFLOPS
- [ ] After register opt: ≥720 GFLOPS
- [ ] **Final target: ≥750 GFLOPS** ✅

### Comparativa
- [ ] vs Original: >15% improvement
- [ ] vs Baseline: >100 GFLOPS gain
- [ ] Stability maintained

---

## 📈 Progreso Esperado

| Hito | Duración | Status | Output |
|------|----------|--------|--------|
| 1.1.3.1 LDS optimization | 1.5h | ⏳ | Optimized variant |
| 1.1.3.2 Coalescing tuning | 1.0h | ⏳ | Optimized variant |
| 1.1.3.3 Register refinement | 1.0h | ⏳ | Optimized variant |
| 1.1.3.4 Full validation | 0.5h | ⏳ | Final report |
| **TOTAL** | **4.0h** | ⏳ | **750-800 GFLOPS** |

---

## 🚀 Ejecución

### Paso 1: Crear kernel optimizado base
```bash
# Copy original kernel
cp src/opencl/kernels/gemm_hybrid.cl \
   src/opencl/kernels/gemm_hybrid_opt.cl

# Then incrementally optimize each aspect
```

### Paso 2: Crear scripts de análisis
```bash
# Subtask 1.1.3.1
python3 scripts/analyze_bank_conflicts.py
python3 scripts/compare_lds_variants.py

# Subtask 1.1.3.2
python3 scripts/analyze_coalescing.py

# Subtask 1.1.3.3
python3 scripts/analyze_register_usage.py
```

### Paso 3: Validación
```bash
# Compare kernels
python3 scripts/compare_kernels.py

# Full validation
python3 scripts/validate_optimizations.py

# Generate report
```

### Paso 4: Phase 1 Sign-off
```bash
# Final metrics
# Final report
# Ready for Phase 2
```

---

## 📊 Métricas a Recolectar

### Para cada variante:
- ✅ Compilation time
- ✅ Binary size
- ✅ Register usage
- ✅ LDS usage
- ✅ Performance (GFLOPS)
- ✅ Error (vs NumPy)
- ✅ Stability (CV %)
- ✅ Memory bandwidth %

### Comparativas:
- ✅ vs Original baseline
- ✅ vs Task 1.1.1
- ✅ vs Task 1.1.2
- ✅ vs target 750-800

---

## 📚 Referencias

**Memory Optimization:**
- GCN 4.0 ISA Manual (AMD)
- ROCm Optimization Guide
- NVIDIA CUDA Best Practices (aplicable a OpenCL)

**Patterns:**
- LDS bank conflict avoidance
- Coalescing patterns
- Register allocation strategies

---

## 🏁 Finalización de Phase 1

### Cuando Task 1.1.3 esté complete:

**Phase 1 Metrics:**
- ✅ Baseline: 542 GFLOPS
- ✅ Phase 1 target: 750-800 GFLOPS
- ✅ Improvement: +130-258 GFLOPS (+38-48%)

**Phase 1 Deliverables:**
- ✅ Hybrid kernel with 4 optimizations
- ✅ Python wrapper (production quality)
- ✅ Comprehensive testing
- ✅ Complete documentation
- ✅ Performance analysis

**Ready for Phase 2:**
- ✅ Sparse matrix formats
- ✅ Advanced kernels
- ✅ Target: 900-1000 GFLOPS

---

## 📞 Soporte

**Si hay errores de compilación:**
1. Revisar gemm_hybrid_opt.cl sintaxis
2. Comparar con gemm_hybrid.cl original
3. Verificar #define macros

**Si performance no mejora:**
1. Analizar profiling data
2. Revisar memory access patterns
3. Considerar trade-offs entre optimizaciones

**Si tests fallan:**
1. Verificar correctness
2. Revisar error calculations
3. Check alpha/beta handling

---

**Status:** 🟡 EN PROGRESO  
**Próximo:** Crear kernel optimizado  
**Deadline:** 4 horas (Phase 1 completion)
