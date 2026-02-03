# Benchmarking - Phase 2, Técnica 1: Block Recursive GEMM

## Resumen de Resultados (24 de enero de 2026)

- **Peak Performance alcanzada:** 91.9 GFLOPS (2048×2048)
- **Performance promedio:** 67.9 GFLOPS
- **Mejora vs Phase 1:** -89.0% (aún por debajo del baseline)
- **Target de la fase:** 860 GFLOPS (no alcanzado)
- **Error numérico:** < 1e-6 en todos los tamaños (excelente correctitud ✅)
- **CV%:** 5-8% (aceptable)

| Tamaño | GFLOPS | Error | CV% | Status |
|--------|--------|-------|------|--------|
| 256    | 31.0   | 2.98e-07 | 7.39 | ❌     |
| 512    | 63.5   | 4.10e-07 | 5.61 | ❌     |
| 1024   | 85.3   | 5.99e-07 | 8.15 | ❌     |
| 2048   | 91.9   | 8.51e-07 | 5.71 | ❌     |

## Evaluación
- **Correctitud:** ✅ Excelente - error numérico insignificante (< 1e-6)
- **Performance:** Mejorado significativamente (~68 GFLOPS avg vs ~12 GFLOPS anterior), pero aún por debajo del baseline de Phase 1 (775 GFLOPS) y muy lejos del target (860 GFLOPS)
- **Progreso:** De 12 GFLOPS a 92 GFLOPS peak mediante corrección de workgroup size y simplificación del kernel

## Optimizaciones Implementadas
- ✅ Aumento de Tile Size: TS=32 → TS=16 (optimizado para workgroup)
- ✅ Workgroup Size: (32,2) → (16,16) = 256 threads (máximo para Polaris)
- ✅ Vectorización Float4: Corregida eliminando vload4
- ✅ Simplificación del Kernel: Cada thread procesa 1 elemento (más eficiente)
- ❌ Performance: Requiere optimizaciones adicionales para alcanzar target

## Corrección del Bug de Vectorización
**Problema:** Uso de `vload4()` causaba error constante ~1.23x en todos los cálculos.

**Solución:** Reemplazar `vload4()` con construcción manual de `float4`:
```c
// Antes (con bug):
float4 a_val = (global_k + 3 < K) ? vload4(0, &A[...]) : (float4)(...);

// Después (corregido):
float4 a_val = (float4)(A[...], (global_k + 1 < K) ? A[...] : 0.0f, ...);
```

**Resultado:** Error reducido de 1.23x a < 1e-6 ✅

## Optimización de Workgroup Size
**Problema:** Workgroup (32,2)=64 threads insuficiente para TS=32.

**Solución:** Cambiar a workgroup (16,16)=256 threads con TS=16:
- Más threads trabajando en paralelo
- Mejor occupancy de la GPU
- Patrón de acceso a memoria más eficiente

**Resultado:** Rendimiento aumentado de ~12 GFLOPS a ~92 GFLOPS ✅

## Recomendación
- **✅ Bugs Corregidos:** Técnica 1 ahora es correcta y tiene rendimiento básico
- **Próximos pasos:** 
  - Optimizaciones avanzadas: async_work_group_copy, prefetching, unrolling
  - Aumentar TS gradualmente con workgroup sizes apropiados
  - Comparar con kernel original de Phase 1 para identificar diferencias restantes
  - Si no se alcanza target, considerar Técnica 2 (Mixed Precision) como alternativa
- **Estado actual:** Base sólida para optimizaciones adicionales

---

_Resultado actualizado tras optimización de workgroup size._
