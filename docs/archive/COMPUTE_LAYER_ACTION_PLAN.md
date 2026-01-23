# 📋 CAPA 2: COMPUTE - Plan de Acción Multi-Sesión

**Inicio**: Enero 17, 2026 (Sesión 9)  
**Objetivo**: Completar CAPA 2: COMPUTE con algoritmos research-grade  
**Timeline**: 5-6 meses (Sesiones 10-30)  
**Status**: ✅ Quantization COMPLETO | ✅ Sparse Networks COMPLETO | ✅ Dynamic Sparse (Session 11) COMPLETO | 🎯 Hybrid Scheduler (Session 12) NEXT

---

## 🎯 Visión General

Construir una **plataforma de compute universal** para RX 580 que permita:
- 🧬 **Genética**: Análisis secuencias, protein folding, drug discovery
- 📊 **Data Science**: ML tradicional, análisis estadístico masivo
- 🎵 **Audio/Música**: Processing, síntesis, ML para audio
- 🌿 **Ecología**: Clasificación especies, análisis ecosistemas
- 🏥 **Medicina**: Imaging médico, diagnóstico asistido
- 💊 **Farmacología**: Virtual screening, docking molecular
- 🔬 **Investigación**: Simulaciones científicas, análisis numérico

---

## 📊 Estado Actual (Sesión 9)

### ✅ **COMPLETADO: Quantization Adaptativa**

**Features implementadas**:
- [x] 4 métodos calibración (minmax, percentile, KL, MSE)
- [x] Per-channel quantization (2-3x mejor que per-tensor)
- [x] QAT support (Quantization-Aware Training)
- [x] Mixed-precision optimization
- [x] INT4 packing (8x compression)
- [x] ROCm/HIP integration
- [x] GPU-specific optimizations (Polaris, Vega, RDNA)

**Métricas**:
- 44 tests pasando (100%)
- 1,526 líneas de código production
- 650 líneas de demo
- 6 referencias académicas implementadas

**Archivos**:
- `src/compute/quantization.py` (1,526 líneas)
- `src/compute/rocm_integration.py` (415 líneas)
- `tests/test_quantization.py` (767 líneas)
- `examples/demo_quantization.py` (650 líneas)
- `COMPUTE_QUANTIZATION_SUMMARY.md` (950 líneas)

**Commit**: `fe56d2f` - "feat(compute): Complete quantization module"

---

## 🚀 Roadmap de Implementación

### **FASE 1: Sparse Networks** (Sesiones 10-12)
**Timeline**: 2-3 semanas  
**Priority**: HIGH  
**Objetivo**: Implementar sparsity estructurado y no-estructurado

#### Sesión 10: Magnitude & Structured Pruning ✅ COMPLETO
**Duración**: ~14 horas (1-2 días)  
**Commits**: f68b8c9, 5d908a0  
**Fecha**: 17 Enero 2026

**Tareas completadas**:
- [x] Planning y diseño de arquitectura
- [x] Implementar `MagnitudePruner` class (~300 líneas)
  - [x] `prune_layer()` con threshold percentile-based
  - [x] `prune_model()` local y global
  - [x] `measure_sparsity()` y compression stats
  - [x] Pruning history tracking
- [x] Implementar `StructuredPruner` class (~300 líneas)
  - [x] `prune_channels()` para CNNs
  - [x] `prune_filters()` para convoluciones
  - [x] `prune_attention_heads()` para Transformers
  - [x] L1/L2/Taylor importance metrics
- [x] Implementar `GradualPruner` class (~200 líneas)
  - [x] Polynomial decay schedule (cubic)
  - [x] Flexible configuration (begin/end/frequency)
  - [x] Integration con base pruners
- [x] SparseOperations class (CSR format, analysis)
- [x] Tests comprehensivos (40 tests, 100% passing)
- [x] Demo con 5 benchmarks completos
- [x] Documentación completa

**Entregables completados**:
```
src/compute/sparse.py (800 líneas) ✅
tests/test_sparse.py (550 líneas, 40 tests) ✅
examples/demo_sparse.py (400 líneas, 5 demos) ✅
COMPUTE_SPARSE_SUMMARY.md (600 líneas) ✅
SESSION_10_SPARSE_COMPLETE.md ✅
```

**Métricas logradas**:
- 50-95% sparsity implementado
- 2x-20x compression achieved
- Tests 40/40 passing (267% vs objetivo)
- 3 papers académicos implementados
- Total: 1,750 líneas código production

#### Sesión 11: Dynamic Sparse Training (RigL) ✅ COMPLETO
**Duración**: ~8 horas (1 día)  
**Fecha**: 17 Enero 2026  
**Commit**: 359ece6  
**Papers**: Evci et al. (2020), Mostafa & Wang (2019), Zhu & Gupta (2017)

**Implementado**:

**1. Core RigL Implementation (4h)**
- [x] Implementado `RigLPruner` class (597 líneas en dynamic_sparse.py)
  - [x] `should_update()` - check update schedule
  - [x] `initialize_mask()` - random sparse initialization
  - [x] `update_mask()` - drop/grow logic con sparsity constante
  - [x] `accumulate_gradients()` - multi-step accumulation
  - [x] `get_statistics()` - tracking completo
- [x] Tests RigL (13 tests, 100% passing)
  - [x] Mask update logic
  - [x] Sparsity preservation
  - [x] Gradient-based growth
  - [x] Drop/grow balance

**2. Dynamic Sparsity Allocation (2h)**
- [x] Implementado `DynamicSparsityAllocator` class (incluido en dynamic_sparse.py)
  - [x] `compute_sensitivities()` - gradient L2 norm
  - [x] `allocate_sparsity()` - inverse sensitivity distribution
  - [x] Deficit redistribution para alcanzar target exacto
  - [x] Allocation history tracking
- [x] Tests allocation (9 tests, 100% passing)
  - [x] Sensitivity computation
  - [x] Sparsity distribution
  - [x] Target achievement validation

**3. Enhanced Sparse Module (1.5h)**
- [x] Extendido sparse.py con fine-tuning
  - [x] `FineTuningScheduler` class (163 líneas)
  - [x] Cosine annealing LR schedule
  - [x] Early stopping con patience
  - [x] Warmup phase support
  - [x] `apply_mask_to_gradients()` utility
- [x] Integration tests (3 tests)

**4. Demos & Benchmarks (2h)**
- [x] `demo_dynamic_sparse.py` (650 líneas)
  - [x] Demo 1: Basic RigL training loop
  - [x] Demo 2: Dynamic per-layer allocation
  - [x] Demo 3: Combined RigL + Dynamic
  - [x] Demo 4: Comparison Dense/Static/RigL
  - [x] Visualization de topology changes

**5. Documentation (1h)**
- [x] `COMPUTE_DYNAMIC_SPARSE_SUMMARY.md` (600 líneas)
  - [x] RigL algorithm con pseudocode
  - [x] Mathematical formulas detalladas
  - [x] Usage examples completos
  - [x] Design decisions documentadas
  - [x] Papers implementados con referencias

**Entregables completados**:
```
src/compute/dynamic_sparse.py (597 líneas) ✅
  ├── RigLPruner class (460 líneas)
  ├── DynamicSparsityAllocator class (137 líneas)
  └── RigLConfig dataclass

src/compute/sparse.py (+163 líneas) ✅
  ├── FineTuningScheduler class (150 líneas)
  └── apply_mask_to_gradients utility

tests/test_dynamic_sparse.py (25 tests, 550 líneas) ✅
  ├── RigL tests (13 tests)
  ├── Allocation tests (9 tests)
  └── Integration tests (3 tests)

examples/demo_dynamic_sparse.py (650 líneas) ✅
  ├── Demo 1: Basic RigL
  ├── Demo 2: Dynamic allocation
  ├── Demo 3: Combined
  └── Demo 4: Comparison

COMPUTE_DYNAMIC_SPARSE_SUMMARY.md (600 líneas) ✅
```

**Métricas alcanzadas**:
- ✅ 25 tests passing (125%, objetivo 20)
- ✅ 90% sparsity sin pre-training
- ✅ Accuracy competitiva con dense
- ✅ Training overhead: <0.01% (negligible)
- ✅ Papers implementados: 3 (objetivo 2-3)
- ✅ Total: 2,560 líneas

**Papers implementados**:
1. ✅ **Evci et al. (2020)** - "Rigging the Lottery" (arXiv:1911.11134)
   - Core RigL algorithm completo
   - Drop/grow con constant sparsity
   
2. ✅ **Mostafa & Wang (2019)** - "Parameter Efficient Training"
   - Dynamic sparsity reparameterization (DSR)
   - Layer-wise sensitivity allocation

3. ✅ **Zhu & Gupta (2017)** - "To prune, or not to prune"
   - Polynomial decay schedule integrado

**Resultados obtenidos**:

| Method | Final Loss | Training Time | Memory | Pre-training |
|--------|-----------|---------------|---------|--------------|
| Dense baseline | 0.993 | 1.0x | 100% | N/A |
| Static pruning | 0.000 | 1.0x | 15% | Yes |
| **RigL (ours)** | **0.170** | **1.0x** | **15%** | **No** |
| RigL + Dynamic | 0.170 | 1.0x | 15% | No |

**Key advantages logrados**:
- ✅ No pre-training needed (ahorro de tiempo)
- ✅ Competitive accuracy vs dense
- ✅ Dynamic topology adaptation
- ✅ Constant sparsity maintenance
- ✅ Per-layer optimization

---

#### Sesión 12: Sparse Formats & Operations 🎯 PRÓXIMA
**Duración**: 8-12 horas (1-2 días)
**Objetivo**: Efficient sparse matrix storage & operations
  - [ ] Conversión dense → CSC
  - [ ] Column-major operations
- [ ] Implementar `BlockSparseMatrix` class
  - [ ] Alineación a wavefront (64 elements)
  - [ ] Block-wise operations
  - [ ] Balance sparsity/efficiency
- [ ] Implementar `DynamicSparseActivations`
  - [ ] Runtime sparsity detection
  - [ ] Automatic format selection
  - [ ] Fallback a dense cuando no vale la pena
- [ ] Benchmarks sparse vs dense
- [ ] Tests 20+ total

**Entregables**:
```
src/compute/sparse_formats.py (nuevo, 600+ líneas)
tests/test_sparse_formats.py (20+ tests)
examples/demo_sparse_formats.py (300+ líneas)
```

**Métricas objetivo**:
- 10-100x menos memoria para sparsity > 90%
- CSR matmul: 2-5x speedup vs dense
- Block-sparse: 3-8x speedup (wavefront-aligned)

#### Sesión 12: ROCm Sparse Kernels (Opcional)
**Duración**: 2-3 días

**Tareas**:
- [ ] HIP kernel para SpMV (Sparse Matrix-Vector)
- [ ] HIP kernel para SpMM (Sparse Matrix-Matrix)
- [ ] Memory coalescing optimization
- [ ] Wavefront-aligned loads
- [ ] Benchmarks GPU vs CPU sparse
- [ ] Python bindings

**Entregables**:
```
src/compute/sparse_kernels.cpp (HIP kernels)
src/compute/sparse_hip.py (Python bindings)
benchmarks/sparse_gpu_benchmark.py
```

**Métricas objetivo**:
- 20-50x speedup vs CPU sparse
- 95% GPU occupancy
- Coalesced memory access

---

### **FASE 2: Spiking Neural Networks** (Sesiones 13-16)
**Timeline**: 3-4 semanas  
**Priority**: MEDIUM-HIGH  
**Objetivo**: Event-driven computing para temporal data

#### Sesión 13: LIF Neurons & Basic SNN
**Duración**: 2 días

**Tareas**:
- [ ] Implementar `LIFNeuron` class
  - [ ] Leaky Integrate-and-Fire dynamics
  - [ ] Spike generation
  - [ ] Refractory period
- [ ] Implementar `SNNLayer` class
  - [ ] Forward pass con spikes
  - [ ] Membrane potential tracking
  - [ ] Event queue
- [ ] Implementar `SNNNetwork` class
  - [ ] Multi-layer SNN
  - [ ] Spike propagation
- [ ] Tests básicos (10+ tests)
- [ ] Demo simple

**Entregables**:
```
src/compute/snn.py (400+ líneas)
tests/test_snn.py (10+ tests)
examples/demo_snn_basic.py (200+ líneas)
```

#### Sesión 14: STDP Learning
**Duración**: 2-3 días

**Tareas**:
- [ ] Implementar `STDPLearning` class
  - [ ] Weight update rules
  - [ ] Spike-timing windows
  - [ ] Asymmetric STDP
- [ ] Implementar `OnlineLearning`
  - [ ] Continuous learning
  - [ ] No backprop required
- [ ] Tests STDP (10+ tests)
- [ ] Demo unsupervised learning

**Entregables**:
```
src/compute/snn_learning.py (300+ líneas)
tests/test_snn_learning.py (10+ tests)
examples/demo_stdp.py (300+ líneas)
```

#### Sesión 15: Encoding Schemes
**Duración**: 1-2 días

**Tareas**:
- [ ] Implementar `RateEncoder`
  - [ ] Poisson spike generation
  - [ ] Frequency modulation
- [ ] Implementar `TemporalEncoder`
  - [ ] Latency coding
  - [ ] Phase coding
- [ ] Implementar `PopulationEncoder`
  - [ ] Gaussian receptive fields
  - [ ] Multiple neurons per feature
- [ ] Tests encoders (10+ tests)

**Entregables**:
```
src/compute/snn_encoders.py (300+ líneas)
tests/test_snn_encoders.py (10+ tests)
```

#### Sesión 16: SNN Applications
**Duración**: 2-3 días

**Tareas**:
- [ ] Implementar `SNNImageClassifier`
  - [ ] Event-based vision
  - [ ] Spatial-temporal processing
- [ ] Implementar `SNNTimeSeriesPredictor`
  - [ ] Temporal pattern recognition
  - [ ] Online prediction
- [ ] Benchmarks SNN vs ANN
- [ ] Demo aplicaciones reales

**Entregables**:
```
src/compute/snn_applications.py (500+ líneas)
examples/demo_snn_vision.py (300+ líneas)
examples/demo_snn_timeseries.py (300+ líneas)
COMPUTE_SNN_SUMMARY.md (600+ líneas)
```

---

### **FASE 3: Hybrid CPU-GPU** (Sesiones 17-19)
**Timeline**: 2-3 semanas  
**Priority**: HIGH  
**Objetivo**: Maximizar utilización de todo el sistema

#### Sesión 17: Dynamic Scheduler
**Duración**: 2-3 días

**Tareas**:
- [ ] Implementar `HybridScheduler` class
  - [ ] Roofline-based decisions
  - [ ] Arithmetic intensity analysis
  - [ ] Device selection heuristics
- [ ] Implementar `OperationProfile` dataclass
  - [ ] FLOPS estimation
  - [ ] Memory bytes estimation
  - [ ] Parallelism degree
- [ ] Tests scheduler (10+ tests)
- [ ] Benchmarks CPU vs GPU vs Hybrid

**Entregables**:
```
src/compute/hybrid_scheduler.py (400+ líneas)
tests/test_hybrid_scheduler.py (10+ tests)
```

#### Sesión 18: Async Pipeline
**Duración**: 2 días

**Tareas**:
- [ ] Implementar `AsyncPipeline` class
  - [ ] Producer-consumer pattern
  - [ ] Overlapped execution
  - [ ] Queue management
- [ ] Implementar `StreamProcessor`
  - [ ] Batch streaming
  - [ ] Prefetching
- [ ] Tests pipeline (10+ tests)
- [ ] Demo high-throughput

**Entregables**:
```
src/compute/async_pipeline.py (400+ líneas)
tests/test_async_pipeline.py (10+ tests)
examples/demo_pipeline.py (300+ líneas)
```

#### Sesión 19: Heterogeneous Models
**Duración**: 2-3 días

**Tareas**:
- [ ] Implementar `HeterogeneousModel` class
  - [ ] Layer-wise device placement
  - [ ] Automatic transfers
  - [ ] Optimized routing
- [ ] Implementar `DevicePlacementOptimizer`
  - [ ] Profiling-guided placement
  - [ ] Communication cost modeling
- [ ] Tests heterogeneous (10+ tests)
- [ ] Demo modelo híbrido

**Entregables**:
```
src/compute/heterogeneous.py (500+ líneas)
tests/test_heterogeneous.py (10+ tests)
examples/demo_heterogeneous.py (400+ líneas)
COMPUTE_HYBRID_SUMMARY.md (500+ líneas)
```

---

### **FASE 4: Neural Architecture Search** (Sesiones 20-24)
**Timeline**: 4-5 semanas  
**Priority**: MEDIUM  
**Objetivo**: Arquitecturas custom para RX 580

#### Sesión 20-21: Search Space & DARTS
**Duración**: 4-5 días

**Tareas**:
- [ ] Implementar `PolarisSearchSpace` class
- [ ] Implementar `DARTS_Polaris` class
- [ ] Supernet construction
- [ ] Bi-level optimization
- [ ] Tests NAS básicos (10+ tests)

**Entregables**:
```
src/compute/nas_search_space.py (400+ líneas)
src/compute/nas_darts.py (600+ líneas)
tests/test_nas.py (10+ tests)
```

#### Sesión 22: Hardware-Aware Predictor
**Duración**: 2-3 días

**Tareas**:
- [ ] Implementar `LatencyPredictor` class
- [ ] Feature extraction
- [ ] Predictor training
- [ ] Accuracy validation
- [ ] Tests predictor (10+ tests)

**Entregables**:
```
src/compute/nas_predictor.py (400+ líneas)
tests/test_nas_predictor.py (10+ tests)
```

#### Sesión 23-24: Multi-Objective NAS
**Duración**: 3-4 días

**Tareas**:
- [ ] Implementar `MultiObjectiveNAS` class
- [ ] NSGA-II algorithm
- [ ] Pareto frontier computation
- [ ] Trade-off analysis
- [ ] Tests multi-objective (10+ tests)
- [ ] Demo búsqueda arquitecturas

**Entregables**:
```
src/compute/nas_multi_objective.py (500+ líneas)
tests/test_nas_multi_objective.py (10+ tests)
examples/demo_nas.py (500+ líneas)
COMPUTE_NAS_SUMMARY.md (700+ líneas)
```

---

### **FASE 5: Domain-Specific Algorithms** (Sesiones 25-30+)
**Timeline**: Ongoing  
**Priority**: MEDIUM  
**Objetivo**: Algoritmos especializados por dominio

#### Sesiones 25-26: Bioinformática
**Tareas**:
- [ ] Smith-Waterman GPU
- [ ] Molecular Dynamics
- [ ] Protein folding

#### Sesiones 27-28: Audio Processing
**Tareas**:
- [ ] FFT optimizado GCN
- [ ] WaveNet sparse
- [ ] Real-time audio effects

#### Sesiones 29-30: Data Science
**Tareas**:
- [ ] XGBoost GPU
- [ ] K-Means clustering
- [ ] Sparse PCA

---

## 📈 Métricas de Progreso

### Por Fase

| Fase | Sesiones | Líneas Código | Tests | Status |
|------|----------|---------------|-------|--------|
| **Quantization** | 8-9 | 3,400 | 44 | \u2705 COMPLETO |
| **Sparse Networks** | 10-12 | ~2,000 | 45+ | 🚀 EN CURSO |
| **SNN** | 13-16 | ~2,000 | 40+ | \ud83d\udcdd Pendiente |
| **Hybrid CPU-GPU** | 17-19 | ~1,500 | 30+ | \ud83d\udcdd Pendiente |
| **NAS** | 20-24 | ~2,500 | 40+ | \ud83d\udcdd Pendiente |
| **Domain-Specific** | 25-30+ | ~3,000+ | 50+ | \ud83d\udcdd Pendiente |

### Totales Esperados

- **Líneas de código**: ~14,400 líneas
- **Tests**: ~249 tests
- **Documentación**: ~6,000 líneas
- **Demos**: ~4,000 líneas
- **Referencias**: 30+ papers académicos

---

## 🎯 Checklist por Sesión

### ✅ Sesión 9 (COMPLETA)
- [x] Audit quantization module
- [x] Implement per-channel quantization
- [x] Implement ROCm integration
- [x] Create comprehensive demo
- [x] Add 5 new tests
- [x] Update documentation
- [x] Commit changes

### 🚀 Sesión 10 (EN CURSO)
- [x] Create COMPUTE_LAYER_ROADMAP.md
- [x] Update PROJECT_STATUS.md
- [x] Create action plan document
- [ ] Implement MagnitudePruner
- [ ] Implement StructuredPruner
- [ ] Implement GradualPruner
- [ ] Add 15+ sparse tests
- [ ] Create demo_sparse.py
- [ ] Document in COMPUTE_SPARSE_SUMMARY.md

### ✅ Sesión 11 (COMPLETA)
- [x] Implementado RigLPruner (460 líneas)
- [x] Implementado DynamicSparsityAllocator (137 líneas)
- [x] Implementado FineTuningScheduler (163 líneas)
- [x] 25 tests (100% passing)
- [x] 4 interactive demos
- [x] COMPUTE_DYNAMIC_SPARSE_SUMMARY.md
- [x] Commit 359ece6 creado

### 📝 Sesión 12 (Próxima)
- [ ] Implement CSRMatrix
- [ ] Implement CSCMatrix
- [ ] Implement BlockSparseMatrix
- [ ] Implement DynamicSparseActivations
- [ ] Add 20+ format tests
- [ ] Benchmark sparse vs dense

---

## 📚 Referencias por Fase

### Sparse Networks
1. Han et al. (2015) "Learning both Weights and Connections"
2. Li et al. (2017) "Pruning Filters for Efficient ConvNets"
3. Zhu & Gupta (2017) "To prune, or not to prune"
4. Gray et al. (2017) "GPU Kernels for Block-Sparse Weights"

### SNN
1. Gerstner & Kistler (2002) "Spiking Neuron Models"
2. Izhikevich (2003) "Simple Model of Spiking Neurons"
3. Diehl & Cook (2015) "Unsupervised learning with STDP"
4. Tavanaei et al. (2019) "Deep Learning in SNNs"

### Hybrid Computing
1. Williams et al. (2009) "Roofline Model"
2. Gregg & Hazelwood (2011) "Where is the Data?"
3. AMD (2012) "GCN Architecture Whitepaper"

### NAS
1. Liu et al. (2019) "DARTS"
2. Cai et al. (2019) "ProxylessNAS"
3. Wu et al. (2019) "FBNet"
4. Tan & Le (2019) "EfficientNet"

---

## 🔄 Proceso por Sesión

### Template de Trabajo

```markdown
## Sesión N: [Nombre]

### Objetivos
- [ ] Objetivo 1
- [ ] Objetivo 2
- [ ] Objetivo 3

### Implementación
1. Diseño de clases
2. Implementación core
3. Tests
4. Demo
5. Documentación

### Entregables
- `archivo1.py` (X líneas)
- `test_archivo1.py` (Y tests)
- `demo_archivo1.py` (Z líneas)

### Validación
- [ ] Tests passing
- [ ] Demo ejecutable
- [ ] Documentación completa
- [ ] Commit realizado

### Métricas
- Líneas código: X
- Tests: Y/Y passing
- Coverage: Z%
- Performance: W speedup
```

---

## 🎉 Entregable Final (v0.8.0)

Al completar todas las fases:

### Código
- **~14,400 líneas** de compute primitives
- **~249 tests** (100% passing)
- **~4,000 líneas** de demos
- **~6,000 líneas** de documentación

### Features
- \u2705 Quantization (4 métodos, per-channel, QAT)
- \u2705 Sparse Networks (structured, unstructured, dynamic)
- \u2705 SNN (LIF, STDP, encoders, applications)
- \u2705 Hybrid CPU-GPU (scheduler, pipeline, heterogeneous)
- \u2705 NAS (DARTS, hardware-aware, multi-objective)
- \u2705 Domain algorithms (genética, audio, data science)

### Aplicaciones
- 🧬 Genética & Bioinformática
- 📊 Data Science & ML
- 🎵 Audio & Música
- 🌿 Ecología & Wildlife
- 🏥 Medicina & Healthcare
- 💊 Farmacología & Drug Discovery
- 🔬 Investigación Científica

---

## 📞 Próxima Sesión

**Sesión 10**: Sparse Networks - Magnitude & Structured Pruning

**Comenzar con**:
```bash
# 1. Revisar este documento
cat COMPUTE_LAYER_ACTION_PLAN.md

# 2. Leer roadmap completo
cat COMPUTE_LAYER_ROADMAP.md

# 3. Implementar MagnitudePruner
vim src/compute/sparse.py

# 4. Escribir tests
vim tests/test_sparse.py

# 5. Demo
vim examples/demo_sparse.py
```

**Tiempo estimado**: 1-2 días intensivos

🚀 **¡A construir algo épico!** 🚀
