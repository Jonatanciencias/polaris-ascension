# Informe Comparativo Final - Estado del Proyecto (7 de febrero de 2026)

## 1) Resumen Ejecutivo
El proyecto está funcional y estable en su ruta principal de producción OpenCL para RX 590.

Resultados clave de esta evaluación exhaustiva:
- Validación funcional: `test_production_system.py` = **4/4 PASS**.
- Suite de tests: `pytest tests/` = **69 passed**.
- Rendimiento reproducible extendido (20 sesiones x 20 iteraciones):
  - **1400x1400**: 779.1 GFLOPS (peak mean)
  - **2048x2048**: 774.9 GFLOPS (peak mean)
  - **512x512**: 471.7 GFLOPS (peak mean)
- Barrido multi-tamaño (3 sesiones x 10 iteraciones) muestra pico observado de **805.2 GFLOPS** en entorno 3072x3072 con `tile24`.

Conclusión: el sistema cumple objetivos de estabilidad y rendimiento reproducible, con rutas legacy puntuales que requieren reparación.

## 2) Alcance y Metodología
### Batería ejecutada
1. Salud general del stack:
   - `./venv/bin/python scripts/verify_hardware.py`
   - `./venv/bin/python scripts/diagnostics.py`
2. Validación funcional principal:
   - `./venv/bin/python test_production_system.py`
   - `./venv/bin/pytest tests/ -v`
3. Rendimiento producción (reproducible):
   - `./venv/bin/python scripts/benchmark_phase3_reproducible.py --sessions 20 --iterations 20 --output-json /tmp/phase3_reproducible_exhaustive_20260207.json`
4. Barrido comparativo por tamaño (`tile20` vs `tile24`) con script ad-hoc:
   - salida: `/tmp/gemm_multisize_sweep_20260207.json`
5. Rendimiento ruta engine/CLI:
   - `./venv/bin/python -m src.cli benchmark --size 1024 --iterations 5`
   - script ad-hoc de sweep `OptimizedKernelEngine` (256..2048): `/tmp/optimized_engine_sweep_20260207.json`
6. Benchmark de referencia de `tests/test_opencl_gemm.py::TestGEMMPerformance`:
   - small/medium/large con salida de GFLOPS.

## 3) Resultados Comparativos
### 3.1 Salud funcional
- `test_production_system.py`: **4/4 PASS**.
  - En la corrida de validación, hardware test reportó:
    - 1400: 779.6 GFLOPS
    - 2048: 774.8 GFLOPS
    - 512: 449.7 GFLOPS
- `pytest tests/ -q`: **69 passed in 11.01s**.
- Hardware detectado:
  - GPU: AMD Radeon RX 590 GME (Clover/Mesa OpenCL 1.1)
  - VRAM: 8 GB

### 3.2 Producción - baseline reproducible extendido (20x20)
Fuente: `/tmp/phase3_reproducible_exhaustive_20260207.json`

| Tamaño | Peak mean | Peak std | Peak min | Peak p95 | Peak max | Avg mean |
|---|---:|---:|---:|---:|---:|---:|
| 1400x1400 | 779.1 | 2.9 | 773.6 | 782.3 | 784.3 | 767.8 |
| 2048x2048 | 774.9 | 1.2 | 772.8 | 776.7 | 778.0 | 712.7 |
| 512x512 | 471.7 | 13.4 | 442.6 | 487.6 | 488.4 | 440.4 |

Lectura:
- 1400 y 2048 muestran estabilidad alta (desviación baja).
- 512 presenta mayor variabilidad relativa (esperable en tamaños pequeños).

### 3.3 Barrido multi-tamaño `tile20` vs `tile24` (3x10)
Fuente: `/tmp/gemm_multisize_sweep_20260207.json`

| Tamaño | Mejor kernel | Peak mean mejor | Peak mean tile20 | Peak mean tile24 |
|---|---|---:|---:|---:|
| 512x512 | tile24 | 441.9 | 214.8 | 441.9 |
| 1024x1024 | tile24 | 708.4 | 425.6 | 708.4 |
| 1300x1300 | tile20 | 787.5 | 787.5 | 722.0 |
| 1400x1400 | tile20 | 775.9 | 775.9 | 741.8 |
| 1800x1800 | tile24 | 782.2 | 747.9 | 782.2 |
| 2048x2048 | tile24 | 777.2 | 294.4 | 777.2 |
| 3072x3072 | tile24 | 804.7 | 169.5 | 804.7 |

Pico observado del barrido:
- **805.2 GFLOPS** (3072x3072, `tile24`, valor máximo de sesión).

### 3.4 Ruta `OptimizedKernelEngine`/CLI (comparativa interna)
Fuentes:
- `/tmp/optimized_engine_sweep_20260207.json`
- `python -m src.cli benchmark --size 1024 --iterations 5`

| Tamaño | Kernel engine | GFLOPS mean | GFLOPS peak |
|---|---|---:|---:|
| 256x256 | gemm_float4_small | 203.1 | 264.5 |
| 512x512 | gemm_float4_vec | 444.3 | 460.9 |
| 1024x1024 | gemm_float4_vec | 528.3 | 529.8 |
| 1400x1400 | gemm_float4_vec | 551.3 | 551.7 |
| 2048x2048 | gemm_float4_vec | 508.0 | 564.0 |

CLI benchmark (1024, 5 iteraciones):
- Mean throughput: **207.0 GFLOPS**
- Peak throughput: **210.5 GFLOPS**

Interpretación:
- La ruta de producción especializada (`tile20`/`tile24`) supera ampliamente la ruta general de engine/CLI.
- Esto es consistente con el diseño: kernels de producción están ajustados para tamaños objetivo.

### 3.5 Benchmark de referencia OpenCL GEMM (tests)
Comando: `pytest -q -s` sobre `TestGEMMPerformance`

- Small 256: **205.69 GFLOPS**
- Medium 512: **445.59 GFLOPS**
- Large 1024: **527.14 GFLOPS**

## 4) Incidencias Detectadas
### 4.1 Scripts legacy rotos
1. `scripts/quick_validation.py`
- Error: `ModuleNotFoundError: No module named 'src.opencl.hybrid_gemm'`

2. `scripts/benchmark_gcn4_optimized.py`
- Error: `ImportError: attempted relative import beyond top-level package`

Impacto: no afectan la ruta principal de producción validada, pero sí degradan cobertura de herramientas auxiliares.

### 4.2 Warnings de entorno (no bloqueantes)
- PyOpenCL cache warning (`%b requires bytes-like object`) durante compilación con cache.
- `clinfo` reporta nota de coexistencia OpenCL 2.2 library / plataformas OpenCL 3.0.

No bloquearon ninguna ejecución de benchmark o test en esta evaluación.

## 5) Conclusión Técnica
Estado actual del proyecto: **operativo y estable** en el camino principal de valor.

- Calidad funcional: **alta** (4/4 + 69/69).
- Rendimiento reproducible producción: **~775-779 GFLOPS** en tamaños clave (1400/2048).
- Pico observado en barrido: **~805 GFLOPS** (condiciones específicas y matriz grande).
- Diferencia clara entre rutas:
  - Producción especializada: máxima performance.
  - Engine/CLI general: menor throughput, útil para operación genérica.

## 6) Recomendaciones Priorizadas
1. P0 - Reparar scripts legacy de validación/benchmark
- Objetivo: recuperar `quick_validation.py` y `benchmark_gcn4_optimized.py`.
- Beneficio: elimina deuda técnica visible en utilidades de diagnóstico.

2. P1 - Unificar benchmark de usuario con kernels de producción
- Objetivo: agregar opción en CLI para benchmark directo `tile20/tile24`.
- Beneficio: evitar discrepancia percibida entre “GFLOPS CLI” y “GFLOPS producción”.

3. P1 - Persistir reportes en `results/`
- Objetivo: guardar automáticamente JSON/Markdown de corridas exhaustivas con timestamp.
- Beneficio: trazabilidad histórica para regresiones de rendimiento.

4. P2 - Mitigar warning de cache PyOpenCL
- Objetivo: revisar compatibilidad de versión/estrategia de compilación cacheada.
- Beneficio: limpieza operativa y menor ruido de diagnóstico.

## 7) Artefactos de Evidencia (esta sesión)
- `/tmp/phase3_reproducible_exhaustive_20260207.json`
- `/tmp/gemm_multisize_sweep_20260207.json`
- `/tmp/optimized_engine_sweep_20260207.json`

## 8) Addendum de Cierre Roadmap Breakthrough (Week 5-6)

Actualización al 7 de febrero de 2026 (cierre formal de roadmap `2026Q1` en rama `feat/breakthrough-roadmap-2026q1`):

- Week 5 - Block 3 (T5 wiring productivo + auto-disable): **promote**
- Week 5 - Block 4 (compatibilidad Rusticl/ROCm): **refine**
- Week 6 - Suite final y cierre de roadmap: **promote**

### Resultado de suite final Week 6
- `test_production_system.py`: **PASS (4/4)**
- `pytest -q tests/`: **74 passed**
- `scripts/validate_breakthrough_results.py`: **6/6 válidos**
- Benchmark productivo (1400, 5x10):
  - `auto`: **900.233 GFLOPS** peak mean
  - `auto_t3_controlled`: **899.357 GFLOPS** peak mean, fallback `0.0`
  - `auto_t5_guarded`: **918.557 GFLOPS** peak mean, overhead ABFT `2.493%`, disable events `0`

### Deuda residual (no bloqueante de cierre)
1. Tests legacy fuera de `tests/` rompen `pytest -q` global por colección.
2. Hardening de selección explícita de plataforma para canary Rusticl (evitar dependencia de `cl.get_platforms()[0]`).
3. Alinear `scripts/verify_drivers.py` con señales reales de `pyopencl`/`clinfo`.

### Evidencia de cierre
- `research/breakthrough_lab/week5_block4_platform_compatibility_decision.json`
- `research/breakthrough_lab/week6_final_closure_decision.json`
- `research/breakthrough_lab/ACTA_WEEK6_FINAL_CLOSURE_2026-02-07.md`

## 9) Addendum Week 8 - Bloques 3/4/5/6 (Actualizacion al 8 de febrero de 2026)

### 9.1 Consolidacion integrada T3/T4/T5 (Block 6)
Fuente: `research/breakthrough_lab/week8_block6_integrated_consolidation_20260208_024445.json`

- Decision global: **promote**
- Week6 suite: **promote**
- T3 drift: **promote** (delta bajo presion `+19.445%`)
- T4 mixed policy: **promote** (reduccion de fallback `0.194`)
- T5 maturation: **promote** (delta uniform recall `+0.017`)
- Auto peak mean en rerun integrado: **901.896 GFLOPS**

### 9.2 Prueba combinada realista T4+T5 (efecto cruzado)
Fuente: `research/breakthrough_lab/week8_block6_t4_t5_interaction_20260208_024510.json`

- Decision: **promote**
- T5 baseline vs combinado:
  - avg GFLOPS: `841.470 -> 843.509` (`+0.242%`)
  - p95 latencia: `13.995 ms -> 13.973 ms` (`-0.159%`)
  - overhead ABFT: `1.143% -> 1.212%` (`+0.069%`)
- T4 en perfil combinado:
  - contract compliance: `1.000`
  - post-fallback violations: `0.000`
  - fallback rate: `0.000`

Lectura:
- No se observa regresion cruzada relevante de latencia/rendimiento en el perfil probado.
- El costo incremental de overhead es pequeno y dentro de guardrails.

### 9.3 Canary corto por plataforma (Clover vs rusticl) en tamanos criticos
Fuente: `research/breakthrough_lab/platform_compatibility/week8_platform_canary_critical_20260208_024625.json`

- Scope:
  - tamanos: `1400`, `2048`
  - kernels: `auto`, `auto_t3_controlled`, `auto_t5_guarded`
- Decision: **promote**
- Correctness max global: `0.0006104` (`<= 1e-3`)
- Ratio minimo rusticl/clover (peak): `0.9229`
- Guardrails T3/T5 en ambas plataformas: pass

Ratios rusticl/clover destacados:
- 1400:
  - auto: `1.009`
  - auto_t3_controlled: `1.012`
  - auto_t5_guarded: `1.013`
- 2048:
  - auto: `0.924`
  - auto_t3_controlled: `0.928`
  - auto_t5_guarded: `0.923`

### 9.4 Gate canonico obligatorio previo al cierre
Fuente: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_024700.json`

- Decision: **promote**
- `pytest tests`: **83 passed**
- schema validation: **green**
- driver smoke JSON: **good**

### 9.5 Conclusión operativa post-Week 8
- El stack de optimizacion (T3/T4/T5) se mantiene estable y promovible con evidencia fresca.
- T4+T5 puede ejecutarse en perfil combinado sin degradacion significativa.
- rusticl se mantiene apto para canary controlado en tamanos criticos, con guardrails activos.

## 10) Addendum Week 9 - Sign-Off Preproduccion (8 de febrero de 2026)

### 10.1 Block 6 - Canary largo de pared + cierre formal
Fuentes:
- `research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_044015.json`
- `research/breakthrough_lab/ACTA_WEEK9_BLOCK6_PREPROD_SIGNOFF_2026-02-08.md`

Resultado:
- Decision Block 6: **promote**
- Wall-clock: **30.0 min** reales (objetivo cumplido)
- Runs OK: **48/48**
- Correctness max: **0.0005646** (`<= 1e-3`)
- Guardrails T3/T5: **pass** (disable T5 = `0`, fallback T3 max = `0.0`)
- Ratio minimo rusticl/clover (peak): **0.9197** (`>= 0.80`)
- Gate canonico posterior: **promote** (`pytest tests` = **85 passed**, smoke drivers JSON = `good`)

### 10.2 Estado comparativo vs baseline original

- Frente al estado base inicial del informe (7-feb), el proyecto pasa de "estable para pruebas controladas" a **candidato de despliegue controlado**, con sign-off formal preproduccion completado.
- La cadena activa Week 9 (Block2..6) queda consolidada en `promote` sin deuda bloqueante para iniciar rollout controlado.

## 11) Addendum Week 10 - Arranque de Rollout Controlado (8 de febrero de 2026)

### 11.1 Block 1 (scope bajo con rollback automatico)
Fuentes:
- `research/breakthrough_lab/platform_compatibility/week10_block1_controlled_rollout_20260208_160122.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_160122.json`
- `research/breakthrough_lab/ACTA_WEEK10_BLOCK1_CONTROLLED_ROLLOUT_2026-02-08.md`

Resultado:
- Decision Block 1: **iterate**
- Snapshots ejecutados: **2/3** (detenido por guardrail)
- Trigger de rollback: **T5 hard guardrail** (disable event en snapshot 2)
- Rollback: **exitoso** y gate canonico posterior **promote**

Lectura:
- El marco de seguridad operacional funciona como se esperaba (deteccion + rollback + validacion).
- Para promover el rollout, queda trabajo puntual de hardening T5 en perfil de bajo alcance.

### 11.2 Dashboard extendido (Block 6 explicito + drift semanal)
Fuentes:
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_160146.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_160146.md`

Mejora aplicada:
- Cadena activa ahora visible como `block2 -> block3 -> block4 -> block5 -> block6 -> block10`.
- Se agrega tracking de drift por transicion de bloque para T3/T5.

### 11.3 Block 1.1 (hardening T5) - cierre de la deuda de disable events
Fuentes:
- `research/breakthrough_lab/platform_compatibility/week10_block1_1_controlled_rollout_20260208_161153.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_161219.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_161230.json`

Resultado:
- Decision Block 1.1: **promote**
- Rollout rerun: **4/4 snapshots** completados
- T5 disable events: **0**
- Rollback automatico: **no activado**
- Overhead T5 max: **1.9597%**
- Gate canonico previo a promocion: **promote**

Lectura:
- El hardening T5 elimina el problema detectado en Week10 Block1 sin sacrificar correctness ni guardrails duros.
- El proyecto queda nuevamente en banda de despliegue controlado estable para pruebas reales progresivas.

### 11.4 Block 1.2 + 1.3 (horizonte extendido y cobertura 2048)
Fuentes:
- `research/breakthrough_lab/platform_compatibility/week10_block1_2_controlled_rollout_20260208_163545.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_3_controlled_rollout_20260208_163829.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163611.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_163857.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_163903.json`

Resultado:
- Block 1.2: **promote** (`6/6` snapshots, `1400`, rollback `false`, T5 disable `0`)
- Block 1.3: **promote** (`6/6` snapshots, `1400+2048`, rollback `false`, T5 disable `0`)
- Gates canonicos previos a promocion: **promote** en ambos bloques
- Dashboard integrado: **promote** con cadena activa estable

Lectura:
- El perfil endurecido mantiene estabilidad temporal y de cobertura sin degradar guardrails.
- Estado actual: listo para continuar con una ventana controlada mas larga antes de escalar alcance operativo.

### 11.5 Block 1.4 + 1.5 (ventana larga + split Clover/rusticl)
Fuentes:
- `research/breakthrough_lab/platform_compatibility/week10_block1_4_long_window_20260208_165345.json`
- `research/breakthrough_lab/platform_compatibility/week10_block1_5_platform_split_20260208_165631.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165410.json`
- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260208_165700.json`
- `research/breakthrough_lab/week9_comparative_dashboard_20260208_165707.json`

Resultado:
- Block 1.4: **promote** (`>=45 min` equivalente, `1400+2048`, rollback `false`)
- Block 1.5: **promote** (split `Clover/rusticl`, ratio minimo `0.9206`)
- Disable events T5 en ambos bloques: **0**
- Gates canonicos antes de promocion: **promote** en ambos bloques

Lectura:
- El sistema mantiene estabilidad bajo ventana larga y split de plataforma con guardrails sanos.
- Se consolida el estado de candidato fuerte para pruebas reales controladas de mayor alcance.
