# Roadmap de Ejecución Inmediata (Week38)

**Fecha**: 2026-02-13  
**Rama objetivo**: `feat/breakthrough-roadmap-2026q1`  
**Objetivo**: Convertir el estado actual en una ejecución continua, con CI estable, PR desbloqueado y operación semanal disciplinada.

---

## 1) Resumen Ejecutivo

Este roadmap prioriza lo pendiente en 3 horizontes:

1. **Ahora (0-24h)**: desbloquear integración (CI + PR + comentarios de revisión).
2. **Corto plazo (24-72h)**: cerrar deuda documental/técnica crítica y dejar baseline operativo semanal.
3. **Week38 (1-2 semanas)**: ejecutar bloques de continuidad post-freeze con evidencia reproducible.

Regla de priorización: **primero estabilidad y trazabilidad**, luego mejoras opcionales.

---

## 2) Backlog Priorizado (Impacto x Urgencia)

### P0 — Debe cerrarse primero

1. **Restaurar estado verde de CI en PR #9**
   - Alcance: jobs fallando en checks principales.
   - Definición de hecho:
     - checks obligatorios en verde;
     - sin fallas nuevas en `cpu-fast`.

2. **Resolver comentarios abiertos de revisión en PR #9**
   - Alcance: hilos sin resolver (código o justificación técnica).
   - Definición de hecho:
     - 0 comentarios pendientes de acción técnica;
     - PR listo para aprobación/rebase final.

3. **Consolidar “fuente de verdad” documental de estado actual**
   - Alcance: evitar contradicciones entre roadmap activo y documentos históricos.
   - Definición de hecho:
     - roadmap activo enlazado desde índice;
     - documentos obsoletos marcados o archivados.

### P1 — Alta prioridad (después de P0)

4. **Subir cobertura útil en rutas críticas (validación/políticas T3/T5)**
   - Alcance: tests que protegen guardrails de rollout.
   - Definición de hecho:
     - nuevas pruebas sobre rutas de fallo real observadas;
     - evidencia de ejecución local.

5. **Alinear metadata de release/versión (README, release notes, empaquetado)**
   - Alcance: narrativa consistente de versión y estado.
   - Definición de hecho:
     - sin discrepancias entre `README.md`, notas de release y metadatos de paquete.

### P2 — Programable (sin bloquear operación)

6. **Mejoras opcionales de performance/auto-tuner de bajo ROI**
   - Solo después de tener operación estable y CI verde.

---

## 3) Plan de Ejecución por Horizonte

## Horizonte A — Arranque inmediato (0-24h)

### A1. Triage y corrección de CI
- Ejecutar validación local equivalente a gate principal.
- Corregir fallas de tests/lint solo relacionadas al PR activo.
- Re-ejecutar hasta obtener resultado verde reproducible.

**Comandos base**

```bash
python scripts/run_validation_suite.py --tier cpu-fast --allow-no-tests --driver-smoke
python -m pytest -q tests/
```

### A2. Limpieza de PR #9
- Revisar todos los comentarios pendientes.
- Aplicar cambios mínimos, trazables y orientados a causa raíz.
- Resolver hilos o documentar “no aplica” con evidencia.

### A3. Cierre de lote A
- Push final de fixes de CI/PR.
- Confirmar checks en verde.

**Salida esperada del Horizonte A**
- PR #9 técnicamente desbloqueado para merge.

---

## Horizonte B — Estabilización (24-72h)

### B1. Cobertura crítica de guardrails
- Añadir/ajustar tests dirigidos en:
  - validación unificada (`scripts/run_validation_suite.py`),
  - políticas T3/T5 con escenarios límite.

### B2. Higiene documental
- Actualizar índice de documentación para apuntar a roadmap vigente.
- Marcar explícitamente documentos de roadmap histórico/obsoleto.

### B3. Alineación de versión/publicación
- Revisar consistencia entre:
  - `README.md`,
  - `RELEASE_NOTES_v2.2.0.md`,
  - `pyproject.toml`/`setup.py`.

**Salida esperada del Horizonte B**
- Base estable para operación semanal sin deuda crítica inmediata.

---

## Horizonte C — Week38 continuidad post-freeze (1-2 semanas)

### C1. Block1 — Monitoreo rutinario controlado
- Ejecutar canario semanal con política vigente.
- Generar artefactos JSON/MD en esquema estándar.
- Decisión formal: `promote` o `iterate` basada en gate canónico.

### C2. Block2 — Robustez bajo presión moderada
- Repetir canario con variación de carga/cola.
- Verificar guardrails T5 y estabilidad de selector.

### C3. Block3 — Consolidación y paquete operativo
- Dashboard comparativo semanal.
- Acta y decisión de continuidad para siguiente ciclo.

**Salida esperada del Horizonte C**
- Cadencia semanal establecida con evidencia y decisiones trazables.

---

## 4) Tablero de Trabajo (Lista Ejecutable)

### Hoy (checklist)
- [ ] Ejecutar `cpu-fast` local y capturar resultado.
- [ ] Corregir causas raíz de checks fallando en PR #9.
- [ ] Resolver comentarios abiertos de revisión.
- [ ] Push final y validar checks remotos.

### Próximas 72h
- [ ] Reforzar tests críticos T3/T5 + validación unificada.
- [ ] Alinear documentación de roadmap activo.
- [ ] Verificar consistencia de versión/release.

### Próxima semana (Week38)
- [ ] Ejecutar Block1 con artefactos y acta.
- [ ] Ejecutar Block2 con presión controlada.
- [ ] Ejecutar Block3 y consolidar decisión semanal.

---

## 5) Criterios de Éxito

Se considera roadmap “encarrilado” cuando:

1. PR #9 queda sin bloqueos técnicos (checks + revisión).
2. La validación local y CI principal convergen en el mismo resultado.
3. Week38 produce artefactos/acta/decisión sin ruptura de guardrails.
4. El equipo puede repetir la cadencia semanal sin trabajo reactivo excesivo.

---

## 6) Riesgos y Mitigación

- **Riesgo**: reaparece drift en T5 bajo ventanas largas.  
  **Mitigación**: canario escalonado + rollback explícito + umbrales hard.

- **Riesgo**: deuda documental vuelve a desalinear decisiones.  
  **Mitigación**: una fuente de verdad y archivado activo de material histórico.

- **Riesgo**: esfuerzo se desvíe a optimizaciones opcionales temprano.  
  **Mitigación**: política de prioridad estricta P0→P1→P2.
