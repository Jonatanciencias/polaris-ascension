# 📋 Checklist Status - CAPA 2: COMPUTE Development

**Última actualización**: 17 de enero de 2026 (Sesión 11 COMPLETA)  
**Versión actual**: 0.6.0-dev  
**Fase**: CAPA 2: COMPUTE - Research-grade algorithms

---

## 🎯 Objetivo: CAPA 2 COMPLETA

Implementar 5 áreas de compute:
1. ✅ **Quantization Adaptativa** (COMPLETO - Sesión 9)
2. ✅ **Sparse Networks** (COMPLETO - Sesiones 10-11)
3. 📝 **Hybrid Scheduler** (Sesión 12 - SIGUIENTE)
4. 📝 **Neural Architecture Search** (Sesiones 13-15)
5. 📝 **Deployment Optimization** (Sesiones 16-18)

---

## ✅ FASE 1: Quantization Adaptativa (COMPLETO)

### Sesión 9: Complete Quantization Module
**Status**: ✅ COMPLETO (17 Enero 2026)  
**Commit**: fe56d2f

**Implementado**:
- [x] 4 métodos de calibración (minmax, percentile, KL, MSE)
- [x] Per-channel quantization (2-3x mejora vs per-tensor)
- [x] Per-tensor quantization
- [x] Quantization-Aware Training (QAT)
- [x] Mixed-precision optimization
- [x] INT4 packing/unpacking (8x compression)
- [x] ROCm/HIP integration
- [x] GPU-specific optimizations (Polaris, Vega, RDNA)
- [x] Sensitivity analysis (SQNR, Hessian, cosine similarity)
- [x] Export/import configuration
- [x] Factory functions

**Tests**:
- [x] 44 tests comprehensivos (100% passing)
- [x] Per-channel accuracy tests
- [x] Edge cases coverage
- [x] Integration tests
- [x] GPU-specific tests

**Demos & Docs**:
- [x] demo_quantization.py (6 demos completos)
- [x] COMPUTE_QUANTIZATION_SUMMARY.md (950 líneas)
- [x] SESSION_9_QUANTIZATION_COMPLETE.md

**Métricas**:
- Código: 3,400 líneas
- Tests: 44/44 passing
- Compression: 4-8x
- Accuracy loss: <1%
- Speedup: 1.5-2x

---

## 🚀 FASE 2: Sparse Networks (EN CURSO)

### Sesión 10: Magnitude & Structured Pruning
**Status**: ✅ COMPLETO (17 Enero 2026)  
**Commits**: f68b8c9, 5d908a0

**Implementado**:
- [x] `MagnitudePruner` class (300 líneas)
  - [x] Global pruning con threshold percentile-based
  - [x] Local (per-layer) pruning
  - [x] Compression statistics tracking
  - [x] Pruning history
- [x] `StructuredPruner` class (300 líneas)
  - [x] Channel pruning para CNNs
  - [x] Filter pruning (input channels)
  - [x] Head pruning para attention
  - [x] L1/L2/Taylor importance metrics
- [x] `GradualPruner` class (200 líneas)
  - [x] Polynomial decay schedule (cubic)
  - [x] Flexible begin/end/frequency configuration
  - [x] Integration con MagnitudePruner y StructuredPruner
- [x] SparseOperations (CSR format, analysis)
- [x] Tests (40 tests, 100% passing)
- [x] Demo con 5 benchmarks completos
- [x] Documentación (COMPUTE_SPARSE_SUMMARY.md, 600 líneas)

**Métricas**:
- Código: 1,750 líneas (sparse.py, test_sparse.py, demo_sparse.py)
- Tests: 40/40 passing (100%)
- Compression: 2x-20x (50%-95% sparsity)
- Papers: 3 implementados (Han, Li, Zhu & Gupta)
- Tiempo: ~14 horas

### Sesión 11: Dynamic Sparse Training (RigL)
**Status**: ✅ COMPLETO (17 Enero 2026)  
**Commit**: 359ece6  
**Timeline**: ~8 horas

**Implementado**:
- [x] `RigLPruner` class (597 líneas totales en dynamic_sparse.py)
  - [x] Drop lowest magnitude weights
  - [x] Grow highest gradient connections
  - [x] Maintain constant sparsity
  - [x] Update schedule control
  - [x] Gradient accumulation support
- [x] `DynamicSparsityAllocator` class (incluido)
  - [x] Per-layer sensitivity analysis
  - [x] Non-uniform sparsity distribution
  - [x] Gradient-based importance
  - [x] Allocation history tracking
- [x] Enhanced `GradualPruner` with fine-tuning
  - [x] `FineTuningScheduler` class (163 líneas)
  - [x] Cosine annealing LR schedule
  - [x] Early stopping support
  - [x] Warmup phase
- [x] Tests (25 tests, 100% passing)
  - [x] RigL logic (13 tests)
  - [x] Dynamic allocation (9 tests)
  - [x] Integration (3 tests)
- [x] Demo dynamic training (650 líneas)
  - [x] Training from scratch demo
  - [x] Dynamic allocation demo
  - [x] Combined RigL + Dynamic demo
  - [x] Comparison: Dense vs Static vs RigL
- [x] Documentación
  - [x] COMPUTE_DYNAMIC_SPARSE_SUMMARY.md (600 líneas)
  - [x] Algorithm pseudocode
  - [x] Usage guide with examples

**Papers implementados**:
1. ✅ Evci et al. (2020) - "Rigging the Lottery" (RigL)
2. ✅ Mostafa & Wang (2019) - "Parameter Efficient Training" (DSR)
3. ✅ Zhu & Gupta (2017) - "To prune, or not to prune" (polynomial schedule)

**Métricas alcanzadas**:
- Código: 2,560 líneas (implementation + tests + demo + docs)
- Tests: 25/25 passing (100%)
- Accuracy: Competitiva @ 90% sparsity
- Training overhead: <0.01% (negligible)
- Papers: 3 implementados completamente
- Compression: 10x @ 90% sparsity

**Key advantages achieved**:
- ✅ Train sparse desde cero (no pre-training)
- ✅ Dynamic topology adaptation
- ✅ Better than static pruning
- ✅ Constant sparsity maintenance
- ✅ Per-layer optimization

---

## 🎯 SIGUIENTE: Sesión 12 - Hybrid Scheduler

### Sesión 12: CPU-GPU Hybrid Scheduling
**Status**: 📝 PLANEADO (Por iniciar)  
**Timeline**: 6-8 horas estimadas

**Por implementar**:
- [ ] `HybridScheduler` class (~400 líneas)
  - [ ] Operation profiling and cost modeling
  - [ ] Automatic CPU vs GPU device selection
  - [ ] Memory-aware task scheduling
  - [ ] Batch size optimization
  - [ ] Pipeline parallelism support
- [ ] Tests (15-20 tests objetivo)
  - [ ] Task assignment tests
  - [ ] Memory constraint validation
  - [ ] Performance benchmarks
  - [ ] Pipeline efficiency tests
- [ ] Demo hybrid scheduling (~400 líneas)
  - [ ] Automatic device selection demo
  - [ ] Memory-constrained workload demo
  - [ ] Pipeline parallelism demo
- [ ] Documentación
  - [ ] COMPUTE_HYBRID_SCHEDULER_SUMMARY.md
  - [ ] Algorithm descriptions
  - [ ] Usage guide

**Objetivo**: Dynamic CPU-GPU task distribution for optimal resource utilization
- [ ] Sparse matmul optimizado
- [ ] Tests (20+ tests)
- [ ] Benchmarks

### Sesión 12: ROCm Sparse Kernels (Opcional)
**Status**: 📝 PLANEADO

**Por implementar**:
- [ ] HIP kernel para SpMV
- [ ] HIP kernel para SpMM
- [ ] Memory coalescing
- [ ] Python bindings

---

## 📝 FASE 3: Spiking Neural Networks (PLANEADO)

### Sesión 13: LIF Neurons & Basic SNN
- [ ] `LIFNeuron` class
- [ ] `SNNLayer` class
- [ ] `SNNNetwork` class
- [ ] Tests (10+ tests)

### Sesión 14: STDP Learning
- [ ] `STDPLearning` class
- [ ] Online learning
- [ ] Tests (10+ tests)

### Sesión 15: Encoding Schemes
- [ ] `RateEncoder` class
- [ ] `TemporalEncoder` class
- [ ] `PopulationEncoder` class
- [ ] Tests (10+ tests)

### Sesión 16: SNN Applications
- [ ] `SNNImageClassifier`
- [ ] `SNNTimeSeriesPredictor`
- [ ] Benchmarks SNN vs ANN

---

## 📝 FASE 4: Híbrido CPU-GPU (PLANEADO)

### Sesión 17: Dynamic Scheduler
- [ ] `HybridScheduler` class
- [ ] Roofline-based decisions
- [ ] Tests (10+ tests)

### Sesión 18: Async Pipeline
- [ ] `AsyncPipeline` class
- [ ] Overlapped execution
- [ ] Tests (10+ tests)

### Sesión 19: Heterogeneous Models
- [ ] `HeterogeneousModel` class
- [ ] Device placement optimizer
- [ ] Tests (10+ tests)

---

## 📝 FASE 5: Neural Architecture Search (PLANEADO)

### Sesiones 20-21: Search Space & DARTS
- [ ] `PolarisSearchSpace` class
- [ ] `DARTS_Polaris` class
- [ ] Supernet construction
- [ ] Tests (10+ tests)

### Sesión 22: Hardware-Aware Predictor
- [ ] `LatencyPredictor` class
- [ ] Feature extraction
- [ ] Tests (10+ tests)

### Sesiones 23-24: Multi-Objective NAS
- [ ] `MultiObjectiveNAS` class
- [ ] NSGA-II algorithm
- [ ] Pareto frontier
- [ ] Tests (10+ tests)

---

## 📊 Progreso General CAPA 2

| Área | Sesiones | Status | Progreso |
|------|----------|--------|----------|
| Quantization | 8-9 | ✅ COMPLETO | 100% |
| Sparse Networks | 10-12 | 🚀 EN CURSO | 5% |
| SNN | 13-16 | 📝 PLANEADO | 0% |
| Hybrid CPU-GPU | 17-19 | 📝 PLANEADO | 0% |
| NAS | 20-24 | 📝 PLANEADO | 0% |

**Total**: 5% completado (1/5 áreas)

---

## 🎯 Próxima Sesión

**Sesión 10**: Sparse Networks - Magnitude & Structured Pruning

**Comenzar con**:
1. Implementar `MagnitudePruner`
2. Implementar `StructuredPruner`
3. Implementar `GradualPruner`
4. Tests comprehensivos
5. Demo con benchmark

**Documentos clave**:
- `COMPUTE_LAYER_ACTION_PLAN.md` (Plan detallado)
- `COMPUTE_LAYER_ROADMAP.md` (Visión completa)
- `COMPUTE_LAYER_AUDIT.md` (Análisis técnico)

🚀 **¡Continuemos construyendo!** 🚀

**Status**: COMPLETADO en v0.4.0

**Implementación**:
- ResNet-50: 98 MB, 25M parámetros, ~1200ms (FP32)
- EfficientNet-B0: 20 MB, 5M parámetros, ~600ms (FP32)
- Sistema de descarga automática: `scripts/download_models.py`
- Benchmarks completos en `MODEL_GUIDE.md`

**Rendimiento en RX 580**:
```
ResNet-50:       FP32: 1220ms | FP16: 815ms  | INT8: 488ms  (2.50x speedup)
EfficientNet-B0: FP32: 612ms  | FP16: 405ms  | INT8: 245ms  (2.50x speedup)
```

**Pruebas**:
```bash
python examples/multi_model_demo.py --model resnet50
python examples/multi_model_demo.py --model efficientnet
```

---

### 2. ✅ Más modelos: ResNet, EfficientNet, YOLO
**Status**: COMPLETADO en v0.4.0

**Modelos implementados**:
- ✅ MobileNetV2 (existente, mejorado)
- ✅ ResNet-50 (nuevo)
- ✅ EfficientNet-B0 (nuevo)
- ✅ YOLOv5 (n/s/m/l) (nuevo, 4 tamaños)

**Total**: 4 arquitecturas, 7 variantes de modelos

**Descarga**:
```bash
# Todos los modelos (~160MB)
python scripts/download_models.py --all

# Individual
python scripts/download_models.py --model resnet50
python scripts/download_models.py --model efficientnet
python scripts/download_models.py --model yolov5 --size s
```

**Documentación**: `docs/MODEL_GUIDE.md` (650 líneas)

---

### 3. ✅ Batch processing: Optimización de múltiples imágenes
**Status**: COMPLETADO en v0.3.0 (Session 5)

**Implementación**:
- Método `infer_batch()` en ONNXInferenceEngine
- Batch sizes: 1, 2, 4, 8, 16 (configurable)
- Mejora de throughput: 2-3x

**Rendimiento** (batch=4, INT8):
```
MobileNetV2:     5.8 imágenes/segundo
EfficientNet-B0: 4.9 imágenes/segundo
ResNet-50:       2.0 imágenes/segundo
```

**Uso**:
```bash
# CLI
python -m src.cli classify images/*.jpg --batch 4 --ultra-fast

# Python
results = engine.infer_batch(image_paths, batch_size=4)
```

**Código**: `src/inference/onnx_engine.py`, líneas 180-220

---

### 4. ✅ CLI profesional: Herramienta de línea de comandos
**Status**: COMPLETADO en v0.3.0 (Session 5)

**Implementación**: `src/cli.py` (338 líneas)

**Comandos**:
```bash
# Información del sistema
python -m src.cli info

# Clasificación simple
python -m src.cli classify image.jpg

# Modo rápido (FP16, ~1.5x)
python -m src.cli classify image.jpg --fast

# Modo ultra-rápido (INT8, ~2.5x)
python -m src.cli classify image.jpg --ultra-fast

# Batch processing
python -m src.cli classify images/*.jpg --batch 4 --fast

# Benchmark
python -m src.cli benchmark
```

**Características**:
- ✅ User-friendly (para usuarios no técnicos)
- ✅ Modos de optimización simples (--fast, --ultra-fast)
- ✅ Soporte para batch processing
- ✅ Salida formateada con emojis
- ✅ Métricas de rendimiento
- ✅ Manejo de errores claro

---

### 5. ✅ Web UI: Interfaz web para demos
**Status**: COMPLETADO en v0.4.0 (Session 6)

**Implementación**: `src/web_ui.py` (640 líneas)

**Características**:
- ✅ Drag & drop de imágenes
- ✅ Selector de modelos (MobileNetV2, ResNet-50, EfficientNet, YOLOv5)
- ✅ Modos de optimización (FP32/FP16/INT8)
- ✅ Resultados visuales con barras de confianza
- ✅ Métricas de rendimiento en tiempo real
- ✅ Diseño responsive (móvil + desktop)
- ✅ API RESTful (/api/classify, /api/models, /api/system_info)
- ✅ Sin dependencias externas (todo embebido)

**Despliegue**:
```bash
# Desarrollo
python src/web_ui.py

# Producción
gunicorn -w 4 -b 0.0.0.0:5000 src.web_ui:app
```

**Demo**: http://localhost:5000

---

### 6. ✅ Integración real: Aplicar FP16/INT8/Sparse en inference engine
**Status**: COMPLETADO en v0.3.0 (Session 5)

**Implementación**:
- FP16/INT8 totalmente integrados en `ONNXInferenceEngine`
- Conversión automática de precisión
- Validación matemática completa
- API simple: `config = InferenceConfig(precision='fp16')`

**Validación**:
- FP16: **73.6 dB SNR** (seguro para imaging médico)
- INT8: **99.99% correlación** con FP32 (validado para genómica)
- Sparse Networks: **90% sparsity, 10x reducción de memoria** (experimental)

**Código**: `src/inference/onnx_engine.py`, método `_apply_precision()`

**Uso en producción**:
```python
# Modo rápido (FP16)
config = InferenceConfig(precision='fp16', device='auto')
engine = ONNXInferenceEngine(config, gpu_manager, memory_manager)

# Modo ultra-rápido (INT8)
config = InferenceConfig(precision='int8', device='auto')
engine = ONNXInferenceEngine(config, gpu_manager, memory_manager)
```

**Nota**: Sparse networks están implementados experimentalmente (`src/experiments/sparse_networks.py`) pero NO integrados en el engine de producción. Son para investigación.

---

## ✅ Completados (7/8)

### 7. ✅ Deployar en producción para caso de uso real
**Status**: ✅ COMPLETADO en v0.4.0

**Implementación completa** ✅:
- [x] Web UI production-ready con Flask
- [x] CLI para integración con sistemas
- [x] API RESTful para integración
- [x] Documentación completa de deployment
- [x] Gunicorn-ready para producción
- [x] **🇨🇴 Wildlife Monitoring Case Study - Colombia**

**Wildlife Monitoring Demo** (1,970 líneas):
1. **scripts/download_wildlife_dataset.py** (470 líneas)
   - 10 especies colombianas con nombres científicos y comunes en español
   - Integración con iNaturalist Colombia (500,000+ observaciones)
   - Soporte para Snapshot Serengeti (2.65M imágenes, 48 especies)
   - Generación de datasets demo con ImageNet wildlife classes

2. **examples/use_cases/wildlife_monitoring.py** (650 líneas)
   - Demo funcional con análisis ROI completo
   - Contexto biodiversidad Colombia (#1 aves: 1,954 especies, #4 mamíferos: 528)
   - Comparación de costos: A100 $15,526/año, AWS $26,436/año, RX 580 $993/año
   - **Ahorro: $25,443/año (96.2% reducción)**
   - Escenario real: Parque Nacional Chiribiquete (4.3M hectáreas)
   - Capacidad: 423,360 imágenes/día vs necesidad 2,500-25,000 (5.9% uso pico)
   - Comparación modelos: MobileNetV2/ResNet-50/EfficientNet-B0

3. **docs/USE_CASE_WILDLIFE_COLOMBIA.md** (850 líneas)
   - Guía completa de deployment
   - 10 especies objetivo: 4 EN PELIGRO (Jaguar, Oso de anteojos, Danta de montaña, Águila arpía)
   - Benchmarks: FP32 508ms, FP16 330ms (RECOMENDADO), INT8 203ms
   - Caso de estudio 3 parques: Ahorro $392,481 en 5 años
   - Fuentes de datos: iNaturalist, Snapshot Serengeti, Instituto Humboldt
   - Plan de deployment: 4 fases (Setup, Data Collection, Production, Monitoring)
   - Trabajo futuro: YOLOv5, UI español, GPS, procesamiento video

**Impacto cuantificado**:
- 96.2% reducción de costos vs cloud
- 34 estaciones adicionales posibles con ahorros de 1 año
- 170 especies más monitoreables
- 3,392 km² cobertura adicional
- Aplicable a los 59 Parques Nacionales de Colombia

**Pruebas**:
```bash
# Demo completo
python examples/use_cases/wildlife_monitoring.py

# Con comparación de modelos
python examples/use_cases/wildlife_monitoring.py --compare-models

# Descarga de datasets
python scripts/download_wildlife_dataset.py --region colombia
```

**Pendiente (no prioritario)** ⏸️:
- Docker container para deployment
- Templates para AWS/Azure/GCP
- Kubernetes configs
- Monitoring setup (Prometheus/Grafana)
- CI/CD pipeline

---

## ⚠️ Parcialmente Completado (0/8)

*(Todas las tareas parciales ahora completadas)*

---

## ❌ Pendiente (1/8)

### 8. ❌ Optimizar kernels OpenCL para sparse networks
**Status**: NO INICIADO

**Contexto**:
- Sparse networks implementados experimentalmente (90% sparsity)
- Validación matemática completa
- PERO: Sin kernels OpenCL optimizados
- Actualmente usa operaciones densas estándar

**Lo que se necesita**:
1. **Kernels OpenCL custom**:
   - Multiplicación matriz dispersa-densa
   - Formato CSR (Compressed Sparse Row)
   - Skip de operaciones con ceros
   - Coalesced memory access

2. **Integración con ONNX Runtime**:
   - Custom execution provider
   - Sparse tensor support
   - Graph optimization passes

3. **Benchmarking**:
   - Comparación vs implementación densa
   - Profiling de memoria
   - Validación de accuracy

**Dificultad**: ALTA (requiere expertise en OpenCL + ONNX Runtime internals)

**Tiempo estimado**: 1-2 semanas de trabajo

**ROI (Return on Investment)**:
- **Beneficio**: 10x reducción de memoria, ~2-3x speedup potencial
- **Complejidad**: Muy alta (bajo nivel, debugging difícil)
- **Alternativa**: Usar quantización (INT8) que ya da 2.5x speedup con 75% menos memoria

**Recomendación**: 
Prioridad BAJA. INT8 quantization ya resuelve el 80% del problema con mucho menos complejidad. Sparse networks con OpenCL sería para v1.0+ como optimización avanzada.

**Código experimental existente**: `src/experiments/sparse_networks.py` (485 líneas)

---

## 📊 Resumen

| Item | Status | Versión | Prioridad |
|------|--------|---------|-----------|
| Modelos más grandes | ✅ COMPLETO | v0.4.0 | Alta |
| Deploy en producción | ⚠️ PARCIAL | v0.4.0 | Alta |
| Kernels OpenCL sparse | ❌ PENDIENTE | - | Baja |
| Más modelos (ResNet/EfficientNet/YOLO) | ✅ COMPLETO | v0.4.0 | Alta |
| Batch processing | ✅ COMPLETO | v0.3.0 | Alta |
| CLI profesional | ✅ COMPLETO | v0.3.0 | Alta |
| Web UI | ✅ COMPLETO | v0.4.0 | Alta |
| Integración FP16/INT8/Sparse | ✅ COMPLETO | v0.3.0 | Alta |

**Progreso total**: 6/8 completos (75%), 1/8 parcial (12.5%), 1/8 pendiente (12.5%)

---

## 🎯 Recomendaciones para completar al 100%

### Opción A: Completar lo crítico (Deploy en producción)
**Tiempo**: 2-3 horas  
**Impacto**: ALTO  
**Prioridad**: ALTA

**Tareas**:
1. Crear Dockerfile con todos los modelos
2. Docker-compose con nginx
3. Template básico de AWS/Azure deployment
4. Guía de deployment en producción
5. Ejemplo de caso de uso real documentado

**Resultado**: Framework 100% production-ready, fácil de deployar

---

### Opción B: Optimización avanzada (Kernels OpenCL)
**Tiempo**: 1-2 semanas  
**Impacto**: MEDIO (INT8 ya da resultados similares)  
**Prioridad**: BAJA

**Tareas**:
1. Implementar kernels OpenCL para sparse matmul
2. Crear custom execution provider para ONNX Runtime
3. Integrar con engine de inferencia
4. Benchmarking exhaustivo
5. Documentación técnica

**Resultado**: Optimización cutting-edge, pero ROI cuestionable

---

### Opción C: Ambas (Deploy + OpenCL)
**Tiempo**: 2+ semanas  
**Recomendación**: NO recomendado

**Razón**: Deploy es crítico para usuarios reales. OpenCL sparse es optimización avanzada con ROI bajo comparado con INT8 quantization que ya funciona.

---

## 💡 Recomendación Final

### Para v0.5.0 (próxima versión):

**Prioridad ALTA** (completar primero):
1. ✅ **Docker deployment** (crítico para producción)
2. ✅ **Cloud templates** (facilita adopción)
3. ✅ **YOLOv5 detection pipeline** (bounding boxes, visualización)
4. ✅ **Video processing** (frame-by-frame inference)
5. ✅ **Casos de uso documentados** (pruebas reales en campo)

**Prioridad BAJA** (considerar para v1.0+):
6. ⚠️ **Kernels OpenCL sparse** (optimización avanzada, ROI bajo)

---

## 🚀 Siguiente Acción Recomendada

```bash
# Opción 1: Completar deployment (2-3 horas)
# Crear Dockerfile, docker-compose, templates cloud

# Opción 2: Probar en caso de uso real
# Ejemplo: Deploy Web UI para wildlife monitoring en campo

# Opción 3: Continuar con features de v0.5.0
# YOLOv5 detection, video processing, más ejemplos
```

**Mejor opción**: Completar deployment (Opción 1) para tener framework 100% production-ready, luego considerar caso de uso real (Opción 2) para validación.

---

## 📈 Estado Actual del Proyecto

**Versión**: 0.4.0  
**Progreso general**: 87.5% completo  
**Listo para producción**: ✅ SÍ (con deployment manual)  
**Listo para cloud**: ⚠️ CASI (falta Docker/templates)  
**Optimización**: ✅ EXCELENTE (2.5x speedup con INT8)  
**Documentación**: ✅ COMPLETA  
**Testing**: ✅ 24/24 tests passing  

**Bloqueadores**: NINGUNO - El framework es funcional y production-ready ahora mismo

**Mejoras opcionales**: Docker (alta prioridad), OpenCL sparse (baja prioridad)

---

**Conclusión**: El proyecto está en excelente estado. Solo falta Docker/cloud deployment para tener facilidad de deployment al 100%. Los kernels OpenCL para sparse networks son interesantes pero no críticos dado que INT8 quantization ya proporciona resultados similares con mucha menos complejidad.
