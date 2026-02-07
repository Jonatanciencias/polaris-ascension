# Acta Week 5 - Block 4 (Compatibilidad Rusticl/ROCm)

- Date: 2026-02-07
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope: reporte formal de factibilidad de plataforma para cierre de Week 5 (`Rusticl/ROCm`) con evidencia reproducible y decision de ruta.

## Objetivo

Ejecutar el bloque de compatibilidad de plataforma:
1. levantar evidencia real de OpenCL/driver/ROCm en host objetivo,
2. validar activacion de Rusticl en modo controlado,
3. comparar señal funcional/performance minima vs Clover,
4. emitir decision formal de continuidad para Phase 4/5.

## Implementacion

Runner formal nuevo:
- `research/breakthrough_lab/platform_compatibility/run_week5_platform_compatibility.py`

Artifacts generados:
- `research/breakthrough_lab/platform_compatibility/week5_platform_compatibility_20260207_234905.json`
- `research/breakthrough_lab/platform_compatibility/week5_platform_compatibility_20260207_234905.md`
- `research/breakthrough_lab/platform_compatibility/report.md`

## Ejecucion Formal

Command:
- `./venv/bin/python research/breakthrough_lab/platform_compatibility/run_week5_platform_compatibility.py`

## Resultados

Estado de plataforma:
- Clover default con GPU AMD detectada: `pass`
- Rusticl visible default con GPU activa: `fail` (aparece sin device por defecto)
- Rusticl activable con `RUSTICL_ENABLE=radeonsi`: `pass`
- ROCm tools (`rocminfo`, `rocm-smi`): no instalados en host actual

Microbench controlado (`tile24`, size 1024):
- Clover avg GFLOPS: `641.616`
- Rusticl avg GFLOPS: `658.612`
- Ratio rusticl/clover: `1.026` (dentro de gate minimo >= 0.9)
- Error numerico: sin desviaciones relevantes (`~1.9e-4` to `2.1e-4`)

Hallazgo de hardening pendiente:
- Ruta productiva usa seleccion hardcodeada por indice de plataforma (`cl.get_platforms()[0]`) en `src/benchmarking/production_kernel_benchmark.py`, impidiendo canary controlado por nombre/plataforma.

## Decision Formal

Track `platform_compatibility`: **refine**.

Razonamiento:
- Rusticl demuestra viabilidad tecnica en modo shadow (activable + microbench correcto).
- No esta listo para promocion de politica de plataforma porque el selector productivo no tiene enrutado explicito de backend/plataforma.
- ROCm se mantiene como ruta opcional/no bloqueante para esta etapa en Polaris.

## Estado de Bloque

`Week 5 - Block 4` queda ejecutado con evidencia reproducible y decision formal registrada.
