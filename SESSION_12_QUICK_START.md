# Session 12 Quick Start Guide
**Fecha**: 18 Enero 2026 (preparado el 17)  
**Objetivo**: Sparse Matrix Formats & GPU Operations

---

## 🎯 Objetivo de la Sesión

Implementar formatos de matrices sparse eficientes (CSR, CSC, Block-sparse) optimizados para AMD Radeon RX 580.

---

## 📋 Checklist de Inicio

### 1. Verificar Estado del Proyecto
```bash
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580
git status                    # Debe estar limpio
git log --oneline -3          # Ver últimos commits
PYTHONPATH=. pytest tests/ -q # Verificar 155 tests passing
```

### 2. Leer Documentación Preparatoria
- [ ] `NEXT_STEPS.md` - Plan completo Session 12
- [ ] `COMPUTE_LAYER_ACTION_PLAN.md` (líneas 102-250) - Detalles técnicos
- [ ] `ARCHITECTURE_AUDIT_REPORT.md` - Estado del proyecto

### 3. Revisar Papers de Referencia
- [ ] Gray et al. (2017) "GPU Kernels for Block-Sparse Weights"
- [ ] NVIDIA (2020) "Accelerating Sparse Deep Neural Networks"
- [ ] Buluc et al. (2009) "Parallel Sparse Matrix-Matrix Multiplication"

---

## 🏗️ Plan de Implementación (8-12h)

### Fase 1: CSR Format (3h)
```bash
# Crear archivo
touch src/compute/sparse_formats.py

# Implementar CSRMatrix class
# - Conversion dense → CSR
# - Sparse matmul optimizado
# - Memory footprint analysis
```

**TDD Workflow**:
1. Crear `tests/test_sparse_formats.py`
2. Escribir tests primero (CSR conversion, correctness)
3. Implementar hasta que pasen
4. Refactor

### Fase 2: CSC & Block-Sparse (3h)
```bash
# Extender sparse_formats.py
# - CSCMatrix class
# - BlockSparseMatrix class
# - Wavefront alignment (64 elements para RX 580)
```

### Fase 3: Dynamic Selection (2h)
```bash
# Implementar DynamicSparseActivations
# - Runtime sparsity analysis
# - Automatic format selection
# - Fallback to dense
```

### Fase 4: Demos & Benchmarks (2h)
```bash
touch examples/demo_sparse_formats.py

# Implementar:
# - CSR/CSC comparison
# - Block-sparse demo
# - Performance benchmarks
# - Memory comparison
```

### Fase 5: Documentation (1h)
```bash
touch COMPUTE_SPARSE_FORMATS_SUMMARY.md

# Documentar:
# - Algorithms implemented
# - Benchmark results
# - Usage examples
# - Integration guide
```

---

## 🧪 Testing Strategy

### Unit Tests (15 tests mínimo)
```python
# test_sparse_formats.py structure:
class TestCSRMatrix:
    - test_dense_to_csr_conversion()
    - test_csr_matmul_correctness()
    - test_memory_compression()
    - test_empty_matrix()
    - test_high_sparsity_90_percent()

class TestCSCMatrix:
    - test_dense_to_csc_conversion()
    - test_csc_matmul_correctness()
    - test_column_operations()

class TestBlockSparse:
    - test_block_alignment()
    - test_wavefront_optimization()
    - test_block_matmul()

class TestDynamicSelection:
    - test_format_selection_logic()
    - test_fallback_to_dense()
    - test_runtime_switching()

class TestIntegration:
    - test_integration_with_pruner()
    - test_integration_with_rigl()
```

---

## 📊 Success Metrics

### Minimum Viable (MVP)
- [ ] 20+ tests passing
- [ ] CSR/CSC implemented
- [ ] Correctness validated vs dense
- [ ] Basic documentation

### Target Goals
- [ ] 10-100x memory reduction (sparsity > 90%)
- [ ] 2-5x speedup vs dense (CSR/CSC)
- [ ] 3-8x speedup (Block-sparse)
- [ ] Integration con Sessions 10-11

### Stretch Goals
- [ ] GPU kernels (HIP/ROCm) - Puede quedar para Session 13
- [ ] Automatic format tuning
- [ ] Production-ready performance

---

## 🔗 Código Existente para Integrar

### src/compute/sparse.py
```python
class SparseOperations:
    # Ya tiene:
    - analyze_sparsity()
    - sparse_matmul() [placeholder, mejorar con CSR]
    
    # Integrar con:
    - CSRMatrix para sparse_matmul()
    - Memory analysis con format comparison
```

### src/compute/dynamic_sparse.py
```python
class RigLPruner:
    # Ya genera máscaras sparse:
    mask = pruner.get_mask()  # Binary mask
    
    # Puede usar:
    csr_weights = CSRMatrix.from_dense(weights, mask)
```

### src/core/gpu.py
```python
class GPUManager:
    # Ya tiene info de hardware:
    wavefront_size = 64  # Para RX 580
    
    # Usar para:
    - Block-sparse alignment
    - Optimal block size tuning
```

---

## 💡 Tips & Gotchas

### Performance Considerations
- **Sparsity threshold**: CSR solo beneficia si sparsity > 70-80%
- **Block size**: Múltiplos de 64 para wavefront alignment
- **Memory access**: Coalesced reads importantes para GPU

### Common Pitfalls
- ⚠️ CSR matmul puede ser más lento que dense para baja sparsity
- ⚠️ Conversión dense→CSR tiene overhead, cachear si posible
- ⚠️ Validar correctness con tolerancia (float precision)

### Debug Strategy
```bash
# Si tests fallan:
1. Verificar conversión dense→CSR manualmente
2. Comparar resultado CSR matmul vs NumPy dense
3. Usar matrices pequeñas para debug (4x4, 8x8)
4. Print intermediate steps (indices, values, row_ptr)
```

---

## 🚀 Comandos Rápidos

### Iniciar Session 12
```bash
cd /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580
source venv/bin/activate

# Crear archivos base
touch src/compute/sparse_formats.py
touch tests/test_sparse_formats.py
touch examples/demo_sparse_formats.py

# Primer test
echo "import pytest
import numpy as np
from src.compute.sparse_formats import CSRMatrix

class TestCSRMatrix:
    def test_dense_to_csr(self):
        dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        csr = CSRMatrix.from_dense(dense)
        assert len(csr.values) == 3
        assert csr.nnz == 3
" > tests/test_sparse_formats.py

# Run (fallará, expected)
PYTHONPATH=. pytest tests/test_sparse_formats.py -v
```

### Durante Desarrollo
```bash
# Test continuo
PYTHONPATH=. pytest tests/test_sparse_formats.py -v --tb=short

# Test específico
PYTHONPATH=. pytest tests/test_sparse_formats.py::TestCSRMatrix::test_dense_to_csr -v

# Todos los tests
PYTHONPATH=. pytest tests/ -q
```

### Finalizar Session
```bash
# Verificar todo pasa
PYTHONPATH=. pytest tests/ -q

# Commit
git add -A
git commit -m "Session 12: Sparse Matrix Formats - COMPLETE

✅ Implemented CSR, CSC, Block-sparse formats
✅ 20+ tests passing
✅ Benchmarks show 2-5x speedup
✅ Integration with Sessions 10-11
✅ Documentation complete
"
```

---

## 📚 Referencias Útiles

### NumPy Sparse Operations
```python
# Crear matriz sparse manualmente
import numpy as np
from scipy import sparse

# CSR example
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
```

### RX 580 Hardware Specs
- Wavefront size: 64
- CUs: 36
- Memory bandwidth: 256 GB/s
- Cache L2: 2 MB

---

**Ready to start**: ✅ TODO PREPARADO  
**Estimated time**: 8-12 hours  
**Expected output**: 3 new files, 20+ tests, complete docs

**¡Buena suerte! 🚀**
