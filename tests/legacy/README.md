# Tests Legacy

Este directorio contiene tests que fueron creados para versiones anteriores del proyecto
y que actualmente tienen imports rotos debido a refactorizaciones de la estructura del código.

## Razón de la migración

Estos tests hacen referencia a módulos que ya no existen en la estructura actual:

- `core.gpu`, `core.memory` - Módulos core legacy
- `src.api`, `src.inference`, `src.compute` - APIs antiguas
- `fase_9_breakthrough_integration` - Estructura de fases antigua
- `distributed`, `utils` - Módulos movidos o eliminados

## Estado

- **NO SE EJECUTAN**: Estos tests están excluidos del pipeline de CI
- **ARCHIVADOS**: Se mantienen por referencia histórica
- **NO MANTENIDOS**: No se actualizarán para funcionar con la estructura actual

## Opciones futuras

1. **Eliminar**: Si no se necesitan, pueden eliminarse completamente
2. **Adaptar**: Si la funcionalidad existe, adaptar a nuevos módulos
3. **Referenciar**: Usar como referencia para crear tests nuevos

## Tests movidos aquí

- test_gpu.py - Tests de GPUManager (ahora en OptimizedKernelEngine)
- test_memory.py - Tests de MemoryManager (ahora en AdvancedMemoryManager)
- test_api.py, test_api_cluster.py - API REST antigua
- test_integration.py - Integración de HybridOptimizer antiguo
- test_distributed.py - Sistema distribuido no implementado
- Y otros...

## Fecha de migración

2026-02-03 - Limpieza y sanitización de tests legacy
