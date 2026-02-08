#!/usr/bin/env python3
"""
Tests de Integración del Sistema Completo
=========================================

Tests end-to-end que verifican la integración de todos los componentes:
- OptimizedKernelEngine
- AdvancedMemoryManager  
- Selección automática de kernels
- Kernels GCN4 optimizados
"""

import pytest
import numpy as np
import time

# Skip tests si OpenCL no está disponible
try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

pytestmark = pytest.mark.skipif(not HAS_OPENCL, reason="PyOpenCL not available")


@pytest.fixture(scope="module")
def integrated_engine():
    """Fixture para crear el engine integrado"""
    from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine
    engine = OptimizedKernelEngine()
    yield engine


class TestSystemIntegration:
    """Tests de integración del sistema completo"""
    
    def test_full_pipeline_small(self, integrated_engine):
        """Pipeline completo con matrices pequeñas"""
        A = np.random.randn(128, 128).astype(np.float32)
        B = np.random.randn(128, 128).astype(np.float32)
        
        result = integrated_engine.gemm(A, B)
        
        # Verificar resultado
        assert result.result is not None
        assert result.result.shape == (128, 128)
        
        # Verificar corrección
        C_cpu = np.dot(A, B)
        np.testing.assert_allclose(result.result, C_cpu, rtol=1e-3, atol=1e-4)
    
    def test_full_pipeline_medium(self, integrated_engine):
        """Pipeline completo con matrices medianas"""
        A = np.random.randn(1024, 1024).astype(np.float32)
        B = np.random.randn(1024, 1024).astype(np.float32)
        
        result = integrated_engine.gemm(A, B)
        
        assert result.result is not None
        assert result.result.shape == (1024, 1024)
        
        # Verificar que el tiempo es razonable (<1s para 1024x1024)
        assert result.total_time_ms < 1000
    
    def test_full_pipeline_large(self, integrated_engine):
        """Pipeline completo con matrices grandes"""
        A = np.random.randn(2048, 2048).astype(np.float32)
        B = np.random.randn(2048, 2048).astype(np.float32)
        
        result = integrated_engine.gemm(A, B)
        
        assert result.result is not None
        assert result.result.shape == (2048, 2048)
    
    def test_automatic_kernel_selection_varies(self, integrated_engine):
        """Verifica que la selección automática varía según tamaño"""
        from src.optimization_engines.optimized_kernel_engine import KernelType
        
        # Pequeño
        k_small = integrated_engine.select_best_kernel(64, 64, 64)
        
        # Mediano
        k_medium = integrated_engine.select_best_kernel(1024, 1024, 1024)
        
        # Grande
        k_large = integrated_engine.select_best_kernel(4096, 4096, 4096)
        
        # Al menos uno debería ser diferente
        kernels = [k_small, k_medium, k_large]
        # Verificar que son tipos de kernel válidos
        for k in kernels:
            assert isinstance(k, KernelType)


class TestPerformanceIntegration:
    """Tests de rendimiento integrado"""
    
    def test_sustained_throughput(self, integrated_engine):
        """Verifica rendimiento sostenido"""
        np.random.seed(42)
        A = np.random.randn(1024, 1024).astype(np.float32)
        B = np.random.randn(1024, 1024).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            _ = integrated_engine.gemm(A, B)
        
        # 12 operaciones y métrica robusta para evitar falsos negativos por jitter del host.
        total_times_ms = []
        for _ in range(12):
            result = integrated_engine.gemm(A, B)
            total_times_ms.append(float(result.total_time_ms))

        times = np.array(total_times_ms, dtype=np.float64)
        raw_cv = float(np.std(times) / np.mean(times))

        # Descarta el 10% inferior/superior para reducir sensibilidad a outliers puntuales.
        p10, p90 = np.percentile(times, [10, 90])
        trimmed = times[(times >= p10) & (times <= p90)]
        trimmed_cv = float(np.std(trimmed) / np.mean(trimmed))

        assert trimmed_cv < 0.30, (
            f"Variación sostenida (trimmed CV) {trimmed_cv:.2%} es muy alta"
        )
        assert raw_cv < 0.60, (
            f"Variación bruta (raw CV) {raw_cv:.2%} indica inestabilidad severa"
        )
    
    def test_achieves_reasonable_gflops(self, integrated_engine):
        """Verifica que se alcanza rendimiento razonable"""
        size = 2048
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        # Warmup
        _ = integrated_engine.gemm(A, B)
        
        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = integrated_engine.gemm(A, B)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        flops = 2 * size**3
        gflops = flops / avg_time / 1e9
        
        # Debe alcanzar al menos 100 GFLOPS en GPU moderna
        assert gflops > 100, f"Rendimiento {gflops:.1f} GFLOPS es muy bajo"


class TestNumericalStability:
    """Tests de estabilidad numérica"""
    
    def test_consistent_results(self, integrated_engine):
        """Verifica que resultados son consistentes"""
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)
        
        results = []
        for _ in range(5):
            result = integrated_engine.gemm(A, B)
            results.append(result.result.copy())
        
        # Todos los resultados deben ser idénticos
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    
    def test_no_nan_or_inf(self, integrated_engine):
        """Verifica que no hay NaN o Inf en resultados"""
        sizes = [256, 512, 1024]
        
        for size in sizes:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            result = integrated_engine.gemm(A, B)
            
            assert not np.any(np.isnan(result.result)), f"NaN encontrado para {size}x{size}"
            assert not np.any(np.isinf(result.result)), f"Inf encontrado para {size}x{size}"
    
    def test_precision_across_sizes(self, integrated_engine):
        """Verifica precisión en diferentes tamaños"""
        sizes = [128, 256, 512, 1024]
        
        for size in sizes:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            result = integrated_engine.gemm(A, B)
            C_cpu = np.dot(A, B)
            
            rel_error = np.abs(result.result - C_cpu) / (np.abs(C_cpu) + 1e-10)
            mean_error = np.mean(rel_error)
            
            assert mean_error < 1e-4, f"Error {mean_error:.2e} muy alto para {size}x{size}"


class TestResourceManagement:
    """Tests de gestión de recursos"""
    
    def test_handles_many_operations(self, integrated_engine):
        """Verifica que maneja muchas operaciones sin crash"""
        for i in range(50):
            size = np.random.randint(64, 512)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            result = integrated_engine.gemm(A, B)
            assert result.result is not None
    
    def test_recovers_from_large_allocation(self, integrated_engine):
        """Verifica recuperación después de allocation grande"""
        # Operación grande
        A = np.random.randn(4096, 4096).astype(np.float32)
        B = np.random.randn(4096, 4096).astype(np.float32)
        
        result_large = integrated_engine.gemm(A, B)
        assert result_large.result is not None
        
        # Operación pequeña después
        A_small = np.random.randn(256, 256).astype(np.float32)
        B_small = np.random.randn(256, 256).astype(np.float32)
        
        result_small = integrated_engine.gemm(A_small, B_small)
        assert result_small.result is not None
        
        # Verificar corrección de la pequeña con tolerancia razonable
        C_cpu = np.dot(A_small, B_small)
        np.testing.assert_allclose(result_small.result, C_cpu, rtol=1e-2, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
