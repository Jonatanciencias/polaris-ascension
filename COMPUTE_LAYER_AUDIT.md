# COMPUTE LAYER - Auditoría Técnica y Plan de Implementación

**Fecha**: 16 de enero de 2026  
**Versión**: 0.5.0-dev  
**Estado**: Fase de desarrollo activa

---

## 📊 Estado Actual: Quantization Module

### ✅ Lo que YA existe (Funcional pero básico)

#### 1. **Estructura de clases bien definida**
```python
- QuantizationPrecision (Enum): FP32, FP16, INT8, INT4
- QuantizationConfig (dataclass): Configuración
- LayerQuantizationStats (dataclass): Métricas por capa
- AdaptiveQuantizer (class): Clase principal
```

#### 2. **Métodos implementados**
- ✅ `analyze_layer_sensitivity()`: Análisis básico de sensibilidad
- ✅ `quantize_tensor()`: Quantización INT8/FP16
- ✅ `dequantize_tensor()`: Dequantización
- ✅ `get_optimal_precision()`: Selector de precisión
- ✅ `generate_quantization_report()`: Reporte básico
- ✅ Soporte symmetric/asymmetric quantization

#### 3. **Configuraciones GPU-specific**
```python
- Polaris (RX 580): INT8 recomendado, 8GB VRAM
- Vega 56/64: FP16 con Rapid Packed Math
- Navi: FP16 con aceleración
```

---

## ❌ Lo que FALTA (Research-grade enhancements)

### 1. **Calibración Matemáticamente Rigurosa**

**Problema actual**: Solo usa min/max para calcular scale/zero_point

**Necesario**:
- **KL Divergence Minimization** (TensorRT approach)
  - Encuentra threshold óptimo que minimiza divergencia de distribución
  - Formula: `D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))`
  
- **Percentile-based calibration**
  - Usar P99.9 en lugar de max para evitar outliers
  - Más robusto que min/max simple
  
- **Histograms y binning**
  - Construir histograma de activaciones
  - Optimizar bins para minimizar error de quantización

**Referencias**:
- Migacz, S. (2017). "8-bit Inference with TensorRT" - NVIDIA GTC
- Jacob et al. (2018). "Quantization and Training of Neural Networks"

### 2. **Quantization-Aware Training (QAT)**

**Problema actual**: Solo Post-Training Quantization (PTQ)

**Necesario**:
- **Fake Quantization** durante forward pass
  - `y = fake_quant(x) = dequantize(quantize(x))`
  - Permite gradientes fluir durante backprop
  
- **Straight-Through Estimator (STE)**
  - Formula: `∂L/∂x ≈ ∂L/∂y` (gradiente pasa sin modificar)
  - Permite entrenar con quantización
  
- **Learning rate scheduling**
  - Fine-tuning con LR bajo para convergencia

**Referencias**:
- Bengio et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons"
- Google TensorFlow QAT documentation

### 3. **Sensitivity Analysis Avanzado**

**Problema actual**: Solo usa std como métrica de sensibilidad

**Necesario**:
- **Hessian Trace** (segunda derivada de loss)
  - `Tr(H) = Σ ∂²L/∂w²` → Mide curvatura del loss
  - Capas con alta curvatura son más sensibles
  
- **Fisher Information Matrix**
  - Mide información estadística en parámetros
  - Formula: `F = E[(∂log p/∂θ)(∂log p/∂θ)ᵀ]`
  
- **Per-channel vs per-tensor quantization**
  - Granularidad fina para capas sensitivas

**Referencias**:
- Dong et al. (2019). "HAWQ: Hessian AWare Quantization"
- Banner et al. (2018). "Post-training 4-bit quantization"

### 4. **Mixed-Precision Automático**

**Problema actual**: Precisión uniforme para todo el modelo

**Necesario**:
- **Precision search algorithm**
  - Asignar precisión óptima por capa automáticamente
  - Optimizar: min(latency) subject to accuracy_loss < threshold
  
- **Pareto frontier exploration**
  - Trade-off entre accuracy y speed/memory
  - Multiple Pareto-optimal solutions
  
- **Hardware-aware cost model**
  - Usar roofline model del Core Layer
  - Predecir latencia real en RX 580

**Referencias**:
- Wu et al. (2020). "Integer Quantization for Deep Learning Inference"
- Wang et al. (2019). "HAQ: Hardware-Aware Automated Quantization"

### 5. **Optimizaciones específicas GCN**

**Problema actual**: No aprovecha arquitectura GCN

**Necesario**:
- **Wavefront-aligned quantization**
  - Alinear tensors a múltiplos de 64 (wavefront size)
  - Minimizar bank conflicts en memoria
  
- **VALU instruction optimization**
  - GCN tiene VALU (Vector ALU) para INT operations
  - Emular INT8 ops con multiple FP32 VALU
  
- **Memory coalescing patterns**
  - Acceso secuencial a memoria quantizada
  - Maximizar bandwidth utilization (256 GB/s en RX 580)

**Referencias**:
- AMD GCN Architecture Whitepaper
- ROCm Documentation on Integer Operations

### 6. **INT4 y Sub-byte Quantization**

**Problema actual**: INT4 declarado pero no implementado

**Necesario**:
- **4-bit packing/unpacking**
  - Dos valores INT4 en un byte
  - Bit manipulation eficiente
  
- **Mixed INT8/INT4 strategies**
  - Layers menos sensitivas en INT4
  - Reducción 8x vs FP32
  
- **Group quantization**
  - Quantizar grupos de 128 elementos juntos
  - Balance entre precisión y compresión

**Referencias**:
- Shen et al. (2020). "Q-BERT: Hessian Based Ultra Low Precision Quantization"

### 7. **Métricas y Validación**

**Problema actual**: Solo calcula error promedio

**Necesario**:
- **SQNR (Signal-to-Quantization-Noise Ratio)**
  - Formula: `SQNR = 10*log10(σ²_signal / σ²_noise)`
  - Métrica estándar en quantización
  
- **Cosine similarity** entre outputs
  - `cos(θ) = (A·B)/(||A|| ||B||)`
  - Mide preservación de dirección
  
- **Percentile-based error analysis**
  - P50, P95, P99 del error
  - Detectar outliers problemáticos
  
- **Layer-wise accuracy degradation**
  - Tracking de accuracy por capa
  - Identificar bottlenecks

### 8. **Calibration Dataset Management**

**Problema actual**: No hay sistema de calibración con datos

**Necesario**:
- **Representative dataset sampling**
  - Seleccionar subset representativo (100-1000 samples)
  - Clustering para diversidad
  
- **Activation collection**
  - Hook en cada capa para capturar activaciones
  - Estadísticas min/max/histogram por capa
  
- **Caching y serialización**
  - Guardar scales/zero_points calculados
  - Formato JSON/YAML para portabilidad

---

## 📈 Implementación Propuesta - FASE 1

### Prioridad 1: Calibración Avanzada
```python
class AdvancedCalibrator:
    """Calibración con KL divergence y percentiles."""
    
    def calibrate_kl_divergence(
        self,
        activations: np.ndarray,
        num_bins: int = 2048
    ) -> Tuple[float, int]:
        """
        Encuentra threshold que minimiza KL divergence.
        
        Referencias:
        - TensorRT quantization
        - Migacz (2017)
        """
        pass
    
    def percentile_calibration(
        self,
        tensor: np.ndarray,
        percentile: float = 99.99
    ) -> float:
        """Usa percentiles en lugar de max absoluto."""
        pass
```

### Prioridad 2: Quantization-Aware Training
```python
class FakeQuantize:
    """Operador fake quantization para QAT."""
    
    def forward(self, x):
        """Forward con quantize-dequantize."""
        return self.dequantize(self.quantize(x))
    
    def backward(self, grad):
        """Straight-Through Estimator."""
        return grad  # STE: gradient pasa directo
```

### Prioridad 3: Sensitivity Analysis
```python
class SensitivityAnalyzer:
    """Análisis avanzado de sensibilidad."""
    
    def compute_hessian_trace(
        self,
        layer_weights: np.ndarray,
        loss_fn: callable
    ) -> float:
        """
        Calcula traza del Hessian para medir sensibilidad.
        
        Referencias:
        - Dong et al. (2019) HAWQ
        """
        pass
```

### Prioridad 4: Mixed-Precision Search
```python
class MixedPrecisionOptimizer:
    """Búsqueda automática de precisión óptima por capa."""
    
    def find_optimal_precision_assignment(
        self,
        model: dict,
        accuracy_threshold: float = 0.01,
        memory_budget_gb: float = 8.0
    ) -> Dict[str, QuantizationPrecision]:
        """
        Asigna precisión óptima a cada capa.
        
        Optimiza: min(latency) s.t. accuracy_loss < threshold
        """
        pass
```

---

## 🧪 Test Suite Propuesto

```python
# tests/test_quantization.py

def test_kl_divergence_calibration():
    """Verifica que KL calibration reduce error vs min/max."""
    pass

def test_fake_quantization_gradients():
    """Verifica que gradientes fluyen con STE."""
    pass

def test_hessian_sensitivity():
    """Verifica cálculo de Hessian trace."""
    pass

def test_mixed_precision_pareto():
    """Verifica múltiples soluciones Pareto-optimal."""
    pass

def test_int4_packing():
    """Verifica pack/unpack correcto de INT4."""
    pass

def test_sqnr_metric():
    """Verifica cálculo correcto de SQNR."""
    pass

def test_rx580_specific_optimizations():
    """Verifica optimizaciones para RX 580."""
    pass
```

---

## 📚 Referencias Académicas

1. **Jacob et al. (2018)**  
   "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"  
   CVPR 2018

2. **Migacz (2017)**  
   "8-bit Inference with TensorRT"  
   NVIDIA GTC 2017

3. **Dong et al. (2019)**  
   "HAWQ: Hessian AWare Quantization of Neural Networks With Mixed-Precision"  
   ICCV 2019

4. **Banner et al. (2018)**  
   "ACIQ: Analytical Clipping for Integer Quantization"  
   NeurIPS 2018 Workshop

5. **Bengio et al. (2013)**  
   "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"  
   arXiv:1308.3432

6. **Wu et al. (2020)**  
   "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation"  
   arXiv:2004.09602

7. **Wang et al. (2019)**  
   "HAQ: Hardware-Aware Automated Quantization With Mixed Precision"  
   CVPR 2019

---

## 🎯 Métricas de Éxito (KPIs)

### Para considerar implementation "research-grade":

✅ **Accuracy Preservation**
- < 1% accuracy loss on ImageNet con INT8
- < 3% accuracy loss on ImageNet con INT4

✅ **Memory Reduction**
- 75% reduction (FP32 → INT8)
- 87.5% reduction (FP32 → INT4)

✅ **Speed Improvement**
- 1.5-2x faster inference (memory bandwidth bound)
- Batch size increase 2-4x

✅ **Code Quality**
- 100% test coverage
- Documentación completa con ejemplos
- Referencias académicas en docstrings

✅ **Mathematical Rigor**
- KL divergence implementation
- Hessian-based sensitivity
- Formal error bounds

---

## 🚀 Roadmap de Implementación

### Sesión 1 (Ahora)
- [x] Auditoría completa ← **DONE**
- [ ] Implementar calibración KL divergence
- [ ] Implementar percentile-based calibration
- [ ] Tests para calibración

### Sesión 2 (Siguiente)
- [ ] Implementar QAT con fake quantization
- [ ] Straight-Through Estimator
- [ ] Tests para gradientes

### Sesión 3
- [ ] Sensitivity analysis (Hessian trace)
- [ ] Mixed-precision optimizer
- [ ] Tests de sensibilidad

### Sesión 4
- [ ] INT4 implementation completa
- [ ] GCN-specific optimizations
- [ ] Performance benchmarks

---

**Status**: AUDIT COMPLETE ✅  
**Next Action**: Comenzar implementación de calibración avanzada
