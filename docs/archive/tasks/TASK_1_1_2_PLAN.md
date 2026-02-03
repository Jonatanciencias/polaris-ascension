# Task 1.1.2: Implementar Kernel Base - Plan Detallado

**Status:** 🟡 IN PROGRESS  
**Fecha:** 2026-01-24  
**Duración Estimada:** 8 horas (Día 1-2 del Sprint)  
**Prioridad:** CRÍTICA

---

## 📋 Resumen Ejecutivo

**Objetivo:** Compilar el kernel OpenCL y validar su funcionamiento correcto.

**Deliverables:**
1. Kernel compilado sin errores
2. Tests funcionales pasados
3. Métricas de rendimiento iniciales
4. Validación de patrones de memoria

**Criterios de Éxito:**
- ✅ Compilación exitosa sin warnings críticos
- ✅ Exactitud numérica: error < 1e-4
- ✅ Estabilidad: <1% varianza
- ✅ Performance > 600 GFLOPS (baseline mínimo)

---

## 🎯 Desglose de Tareas (8 horas)

### Día 1: Compilación y Validación Rápida (4 horas)

#### Task 1.1.2.1: Validar Compilación (2 horas)

**Objetivo:** Asegurar que el kernel compila sin errores o warnings críticos

**Pasos:**
1. ✅ Verificar dependencias de PyOpenCL
2. ✅ Compilar kernel con configuración default
3. ✅ Documentar cualquier warning
4. ✅ Crear log de compilación

**Archivos:**
- `src/opencl/kernels/gemm_hybrid.cl` (source)
- `src/opencl/hybrid_gemm.py` (compilador)
- `logs/compilation_log.txt` (output)

**Comando:**
```bash
python3 scripts/compile_hybrid_kernel.py --verbose 2>&1 | tee logs/compilation_log.txt
```

**Métricas esperadas:**
- Tiempo de compilación: 2-5 segundos
- Tamaño del binario: 50-100 KB
- Warnings: 0-2 (expected)

#### Task 1.1.2.2: Tests Funcionales Rápidos (2 horas)

**Objetivo:** Validar que el kernel produce resultados correctos

**Pasos:**
1. ✅ Test con matriz pequeña (n=128)
2. ✅ Test con matriz mediana (n=512)
3. ✅ Verificar error vs referencia NumPy
4. ✅ Documentar resultados

**Tests a ejecutar:**
```python
# Test 1: Correctness basic
n = 128
A = np.random.randn(n, n).astype(np.float32)
B = np.random.randn(n, n).astype(np.float32)
C_gpu = executor(A, B)
C_ref = A @ B
error = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
assert error < 1e-4, f"Error too large: {error}"

# Test 2: Alpha/Beta parameters
C_gpu = executor(A, B, alpha=2.5, beta=0.0)
C_ref = 2.5 * (A @ B)
error = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
assert error < 1e-4, f"Alpha test failed: {error}"

# Test 3: Larger matrix
n = 512
A = np.random.randn(n, n).astype(np.float32)
B = np.random.randn(n, n).astype(np.float32)
C_gpu = executor(A, B)
C_ref = A @ B
error = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
assert error < 1e-4, f"Larger matrix test failed: {error}"
```

**Archivo:**
- `scripts/quick_validation.py` (nuevo)

**Comando:**
```bash
python3 scripts/quick_validation.py
```

**Criterios:**
- Error < 1e-4 para todos los tests
- Tiempo < 5 segundos por test
- Sin excepciones no manejadas

---

### Día 2: Benchmarking y Optimización Base (4 horas)

#### Task 1.1.2.3: Performance Baseline (2 horas)

**Objetivo:** Medir GFLOPS iniciales y crear baseline

**Pasos:**
1. ✅ Benchmark con tamaños: 256, 512, 1024, 2048
2. ✅ 10 iteraciones por tamaño
3. ✅ Calcular estadísticas (media, desv. est.)
4. ✅ Comparar vs baseline 542 GFLOPS
5. ✅ Generar gráficos

**Benchmark Code:**
```python
def benchmark_suite():
    sizes = [256, 512, 1024, 2048]
    
    print("Benchmarking Hybrid GEMM Kernel")
    print("-" * 80)
    print(f"{'Size':>6} {'Time (ms)':>12} {'GFLOPS':>10} "
          f"{'Error':>12} {'Speedup vs Base':>15}")
    print("-" * 80)
    
    baseline_gflops = 542
    
    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            C_gpu = executor(A, B)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * size**3) / (mean_time/1000) / 1e9
        
        # Verify accuracy
        C_ref = A @ B
        error = np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref)
        
        speedup = gflops / baseline_gflops
        
        print(f"{size:6d} {mean_time:7.3f}±{std_time:5.3f} "
              f"{gflops:8.1f}  {error:11.2e}  {speedup:13.2f}x")
```

**Archivo:**
- `scripts/benchmark_baseline.py` (nuevo)

**Comando:**
```bash
python3 scripts/benchmark_baseline.py | tee results/baseline_benchmark.txt
```

**Expectativas:**
- n=1024: 600-700 GFLOPS
- Error: < 1e-4
- Speedup vs baseline: 1.0-1.1x (similar al baseline actual)

#### Task 1.1.2.4: Memory Access Analysis (2 horas)

**Objetivo:** Analizar patrones de acceso a memoria y confirmar coalescencia

**Pasos:**
1. ✅ Analizar transacciones de memoria (teórico)
2. ✅ Verificar coalescing en loads float4
3. ✅ Estimación de bandwidth utilizado
4. ✅ Identificar posibles mejoras

**Análisis Teórico:**
```python
def analyze_memory_access(matrix_size=1024):
    tile_size = 16
    
    # Memory transactions
    workgroups_m = (matrix_size + tile_size - 1) // tile_size
    workgroups_n = (matrix_size + tile_size - 1) // tile_size
    workgroups_k = (matrix_size + tile_size - 1) // tile_size
    
    total_workgroups = workgroups_m * workgroups_n
    
    # Each workgroup loads tiles
    # A: tile_size × tile_size = 256 floats
    # B: tile_size × tile_size = 256 floats
    # Total per iteration: 512 floats × 4 bytes = 2 KB
    
    # K iterations
    iterations = workgroups_k
    
    # Total data moved
    data_moved_mb = (total_workgroups * iterations * 2 * 1024) / 1024 / 1024
    
    # Bandwidth required
    time_s = 1.0e-3  # Assume 1ms execution
    bandwidth_required_gb = data_moved_mb / 1024 / time_s
    
    print(f"Matrix size: {matrix_size}×{matrix_size}")
    print(f"Workgroups: {total_workgroups}")
    print(f"K iterations: {iterations}")
    print(f"Total data moved: {data_moved_mb:.1f} MB")
    print(f"Required bandwidth: {bandwidth_required_gb:.1f} GB/s")
    print(f"Available bandwidth: 256 GB/s")
    print(f"Utilization: {min(bandwidth_required_gb/256*100, 100):.1f}%")
```

**Archivo:**
- `scripts/memory_analysis.py` (nuevo)

**Comando:**
```bash
python3 scripts/memory_analysis.py
```

---

## 📊 Progreso Esperado

| Hito | Duración | Entrada | Salida |
|------|----------|---------|--------|
| **1.1.2.1** Compilación | 2h | Kernel source | Binario compilado |
| **1.1.2.2** Tests funcionales | 2h | Kernel compilado | Validación ✅ |
| **1.1.2.3** Benchmarking | 2h | Kernel validado | Métricas de base |
| **1.1.2.4** Análisis memoria | 2h | Benchmarks | Identificación mejoras |

---

## ✅ Criterios de Aceptación

### Compilación
- [ ] Sin errores de compilación
- [ ] Warnings <5 (no críticos)
- [ ] Compilación < 10 segundos

### Funcionalidad
- [ ] test_correctness(n=128): ✅ PASS
- [ ] test_correctness(n=512): ✅ PASS
- [ ] test_alpha_beta: ✅ PASS
- [ ] Error numérico < 1e-4

### Rendimiento
- [ ] n=1024: > 600 GFLOPS
- [ ] Estabilidad: <1% varianza
- [ ] No regression vs baseline

### Documentación
- [ ] Compilation log guardado
- [ ] Benchmark results documentados
- [ ] Memory analysis completado
- [ ] Issues identificados

---

## 🔧 Comandos de Ejecución

### Validación Completa (recomendado)
```bash
# Paso 1: Compilación
python3 scripts/compile_hybrid_kernel.py --verbose

# Paso 2: Tests rápidos
python3 scripts/quick_validation.py

# Paso 3: Benchmarks
python3 scripts/benchmark_baseline.py

# Paso 4: Análisis de memoria
python3 scripts/memory_analysis.py

# Paso 5: Full test suite (opcional, más lento)
python3 scripts/compile_hybrid_kernel.py --full-test
```

### Individual
```bash
# Solo compilación
python3 -c "from src.opencl.hybrid_gemm import HybridGEMMExecutor; e = HybridGEMMExecutor(); print('✅ Compilación exitosa')"

# Solo tests
python3 -m pytest tests/test_gemm_hybrid.py::HybridGEMMTester::test_correctness -v

# Solo benchmarks
python3 scripts/benchmark_baseline.py
```

---

## 📝 Tracking de Progreso

### Checklist de Implementación

**Día 1: Compilación (4h)**
- [ ] Task 1.1.2.1a: Verificar dependencias PyOpenCL
- [ ] Task 1.1.2.1b: Compilar kernel
- [ ] Task 1.1.2.1c: Documentar compilation log
- [ ] Task 1.1.2.2a: Test n=128
- [ ] Task 1.1.2.2b: Test n=512
- [ ] Task 1.1.2.2c: Test alpha/beta

**Día 2: Análisis (4h)**
- [ ] Task 1.1.2.3a: Benchmark suite
- [ ] Task 1.1.2.3b: Generar gráficos
- [ ] Task 1.1.2.4a: Analizar memory access
- [ ] Task 1.1.2.4b: Identificar bottlenecks
- [ ] Task 1.1.2.4c: Documentar hallazgos

---

## 🚀 Next Steps (Task 1.1.3)

Una vez completada Task 1.1.2:

### Task 1.1.3: Optimización de Patrones de Memoria (4h)
- LDS bank conflict optimization
- Global memory coalescing verification
- Float4 load efficiency tuning
- Barrier placement optimization

**Target:** 700-800 GFLOPS

---

## 📚 Archivos de Referencia

**Diseño:**
- `docs/HYBRID_KERNEL_DESIGN.md` - Especificación técnica

**Kernel:**
- `src/opencl/kernels/gemm_hybrid.cl` - Código OpenCL

**Wrapper:**
- `src/opencl/hybrid_gemm.py` - Interfaz Python

**Tests:**
- `tests/test_gemm_hybrid.py` - Suite de testing

---

## 📞 Soporte

Si hay errores durante la compilación:

1. **Error de PyOpenCL:**
   ```bash
   pip3 install pyopencl numpy
   ```

2. **Error de compilación del kernel:**
   - Revisar `logs/compilation_log.txt`
   - Verificar sintaxis OpenCL
   - Confirmar versión del compilador

3. **Error de ejecución:**
   - Verificar disponibilidad de GPU
   - Revisar dimensiones de entrada
   - Confirmar que matrices son C-contiguous

---

**Status:** 🟡 EN PROGRESO  
**Próximo:** Ejecutar scripts de validación  
**Deadline:** Dentro de 8 horas
