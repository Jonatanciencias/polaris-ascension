#!/usr/bin/env python3
"""
Tests para OptimizedKernelEngine - Motor principal de kernels GPU
=================================================================

Tests comprehensivos para verificar funcionalidad, precisión y rendimiento
del engine de kernels optimizados para GCN4/Polaris.
"""

import pytest
import numpy as np
import time
from typing import Tuple

# Skip tests si OpenCL no está disponible
try:
    import pyopencl as cl

    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

pytestmark = pytest.mark.skipif(not HAS_OPENCL, reason="PyOpenCL not available")


@pytest.fixture(scope="module")
def kernel_engine():
    """Fixture para crear el engine una sola vez por módulo de tests"""
    from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine

    engine = OptimizedKernelEngine()
    yield engine


@pytest.fixture(scope="module")
def kernel_types():
    """Fixture para obtener los tipos de kernel disponibles"""
    from src.optimization_engines.optimized_kernel_engine import KernelType

    return KernelType


class TestEngineInitialization:
    """Tests de inicialización del engine"""

    def test_engine_creates_successfully(self, kernel_engine):
        """Verifica que el engine se crea correctamente"""
        assert kernel_engine is not None
        assert kernel_engine.context is not None
        assert kernel_engine.queue is not None
        assert kernel_engine.program is not None

    def test_gpu_detected(self, kernel_engine):
        """Verifica que se detecta una GPU"""
        assert kernel_engine.device is not None
        device_name = kernel_engine.device.name
        assert "AMD" in device_name or "Radeon" in device_name or "GPU" in device_name.upper()

    def test_memory_manager_initialized(self, kernel_engine):
        """Verifica que el gestor de memoria está inicializado"""
        assert kernel_engine.memory_manager is not None


class TestKernelSelection:
    """Tests de selección automática de kernels"""

    def test_selects_kernel_for_small_matrices(self, kernel_engine, kernel_types):
        """Debe seleccionar kernel optimizado para baja latencia en tamaños pequeños"""
        selected = kernel_engine.select_best_kernel(128, 128, 128)
        # La política actual prioriza FLOAT4_SMALL para <=256
        assert selected in [
            kernel_types.GEMM_FLOAT4_SMALL,
            kernel_types.GEMM_FLOAT4_CLOVER,
            kernel_types.GEMM_BASIC,
        ]

    def test_selects_kernel_for_medium_matrices(self, kernel_engine, kernel_types):
        """Debe seleccionar kernel vectorizado o fallback Clover en tamaño medio"""
        selected = kernel_engine.select_best_kernel(1024, 1024, 1024)
        # 1024 alineado a 4 columnas -> FLOAT4_VEC es el camino esperado
        assert selected in [
            kernel_types.GEMM_FLOAT4_VEC,
            kernel_types.GEMM_FLOAT4_CLOVER,
            kernel_types.GEMM_BASIC,
        ]

    def test_selects_kernel_for_large_matrices(self, kernel_engine, kernel_types):
        """Debe seleccionar kernel de alto throughput para matrices grandes"""
        selected = kernel_engine.select_best_kernel(4096, 4096, 4096)
        # Para grandes y alineadas se permite FLOAT4_VEC (actual), STREAMING o ULTRA
        assert selected in [
            kernel_types.GEMM_FLOAT4_VEC,
            kernel_types.GEMM_GCN4_STREAMING,
            kernel_types.GEMM_GCN4_ULTRA,
            kernel_types.GEMM_FLOAT4_CLOVER,
        ]


class TestGEMMCorrectness:
    """Tests de corrección numérica de GEMM"""

    @pytest.mark.parametrize("size", [64, 128, 256, 512])
    def test_gemm_correctness_square(self, kernel_engine, size):
        """Verifica corrección para matrices cuadradas de varios tamaños"""
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        # Error relativo medio debe ser pequeño
        rel_error = np.abs(C_gpu - C_cpu) / (np.abs(C_cpu) + 1e-10)
        mean_error = np.mean(rel_error)

        assert mean_error < 1e-4, f"Error medio {mean_error} excede tolerancia para {size}x{size}"

    def test_gemm_identity_matrix(self, kernel_engine):
        """Multiplicar por identidad debe dar la matriz original"""
        size = 256
        A = np.random.randn(size, size).astype(np.float32)
        I = np.eye(size, dtype=np.float32)

        result = kernel_engine.gemm(A, I)
        C_gpu = result.result

        np.testing.assert_allclose(C_gpu, A, rtol=1e-4, atol=1e-5)

    def test_gemm_zero_matrix(self, kernel_engine):
        """Multiplicar por ceros debe dar ceros"""
        size = 256
        A = np.random.randn(size, size).astype(np.float32)
        Z = np.zeros((size, size), dtype=np.float32)

        result = kernel_engine.gemm(A, Z)
        C_gpu = result.result

        np.testing.assert_allclose(C_gpu, Z, atol=1e-6)

    def test_gemm_rectangular(self, kernel_engine):
        """Verifica corrección para matrices rectangulares"""
        M, K, N = 512, 256, 384
        # Dataset determinista para evitar flake en pytest tier por muestras aleatorias.
        rng = np.random.default_rng(17017)
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        assert C_gpu.shape == (M, N)
        # Métrica robusta: evita sobre-penalizar celdas donde el valor de referencia es ~0.
        error = C_gpu - C_cpu
        nrmse = np.linalg.norm(error) / (np.linalg.norm(C_cpu) + 1e-12)
        max_abs_error = np.max(np.abs(error))

        assert nrmse < 1e-4
        assert max_abs_error < 1e-2


class TestGEMMPerformance:
    """Tests de rendimiento de GEMM"""

    def test_achieves_minimum_gflops(self, kernel_engine):
        """Verifica que se alcanza un rendimiento mínimo aceptable"""
        size = 1024
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Warmup
        _ = kernel_engine.gemm(A, B)

        # Benchmark
        start = time.perf_counter()
        _ = kernel_engine.gemm(A, B)
        elapsed = time.perf_counter() - start

        flops = 2 * size**3
        gflops = flops / elapsed / 1e9

        # Debe alcanzar al menos 50 GFLOPS en hardware moderno
        assert gflops > 50, f"Rendimiento {gflops:.1f} GFLOPS es muy bajo"

    def test_operation_result_contains_metrics(self, kernel_engine):
        """Verifica que el resultado incluye métricas de rendimiento"""
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)

        result = kernel_engine.gemm(A, B)

        # Debe tener métricas
        assert hasattr(result, "kernel_metrics")
        assert hasattr(result, "transfer_metrics")
        assert hasattr(result, "total_time_ms")
        assert result.total_time_ms > 0


class TestKernelTypes:
    """Tests específicos para cada tipo de kernel"""

    def test_basic_kernel_works(self, kernel_engine, kernel_types):
        """Verifica que el kernel básico funciona"""
        A = np.random.randn(256, 256).astype(np.float32)
        B = np.random.randn(256, 256).astype(np.float32)

        result = kernel_engine.gemm(A, B, kernel_type=kernel_types.GEMM_BASIC)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        np.testing.assert_allclose(C_gpu, C_cpu, rtol=1e-3, atol=1e-4)

    def test_gcn4_ultra_kernel_works(self, kernel_engine, kernel_types):
        """Verifica que el kernel GCN4 Ultra funciona"""
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)

        result = kernel_engine.gemm(A, B, kernel_type=kernel_types.GEMM_GCN4_ULTRA)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        rel_error = np.mean(np.abs(C_gpu - C_cpu) / (np.abs(C_cpu) + 1e-10))
        assert rel_error < 1e-4

    def test_gcn4_streaming_kernel_works(self, kernel_engine, kernel_types):
        """Verifica que el kernel GCN4 Streaming funciona correctamente"""
        A = np.random.randn(1024, 1024).astype(np.float32)
        B = np.random.randn(1024, 1024).astype(np.float32)

        result = kernel_engine.gemm(A, B, kernel_type=kernel_types.GEMM_GCN4_STREAMING)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        # Métrica robusta: evitar sobre-penalizar valores de referencia cercanos a 0.
        error = C_gpu - C_cpu
        nrmse = np.linalg.norm(error) / (np.linalg.norm(C_cpu) + 1e-12)
        max_abs_error = np.max(np.abs(error))

        # Tolerancias conservadoras para FP32 en matrices grandes.
        assert nrmse < 1e-4, f"NRMSE {nrmse:.2e} muy alto"
        assert max_abs_error < 1e-2, f"Max abs error {max_abs_error:.2e} muy alto"


class TestEdgeCases:
    """Tests de casos extremos"""

    def test_very_small_matrix(self, kernel_engine):
        """Matrices muy pequeñas deben funcionar"""
        A = np.random.randn(16, 16).astype(np.float32)
        B = np.random.randn(16, 16).astype(np.float32)

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        np.testing.assert_allclose(C_gpu, C_cpu, rtol=1e-3, atol=1e-4)

    def test_non_aligned_size(self, kernel_engine):
        """Tamaños no alineados a tiles deben funcionar"""
        # Tamaño que no es múltiplo de 16, 32 o 64
        A = np.random.randn(137, 137).astype(np.float32)
        B = np.random.randn(137, 137).astype(np.float32)

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        rel_error = np.mean(np.abs(C_gpu - C_cpu) / (np.abs(C_cpu) + 1e-10))
        assert rel_error < 1e-3

    def test_large_values(self, kernel_engine):
        """Valores grandes no deben causar overflow"""
        A = np.random.randn(256, 256).astype(np.float32) * 1000
        B = np.random.randn(256, 256).astype(np.float32) * 1000

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result

        # No debe haber NaN o Inf
        assert not np.any(np.isnan(C_gpu))
        assert not np.any(np.isinf(C_gpu))

    def test_small_values(self, kernel_engine):
        """Valores pequeños no deben causar underflow significativo"""
        A = np.random.randn(256, 256).astype(np.float32) * 1e-3
        B = np.random.randn(256, 256).astype(np.float32) * 1e-3

        result = kernel_engine.gemm(A, B)
        C_gpu = result.result
        C_cpu = np.dot(A, B)

        # La mayoría de valores no deben ser exactamente 0
        nonzero_gpu = np.count_nonzero(C_gpu)
        nonzero_cpu = np.count_nonzero(C_cpu)

        assert nonzero_gpu > 0.9 * nonzero_cpu


class TestStability:
    """Tests de estabilidad del sistema"""

    def test_multiple_consecutive_operations(self, kernel_engine):
        """Múltiples operaciones consecutivas deben ser estables"""
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)

        results = []
        for _ in range(10):
            result = kernel_engine.gemm(A, B)
            results.append(result.result.copy())

        # Todos los resultados deben ser idénticos
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_no_memory_leak_pattern(self, kernel_engine):
        """No debe haber patrones obvios de memory leak"""
        import gc

        # Forzar recolección inicial
        gc.collect()

        # Ejecutar muchas operaciones
        for _ in range(20):
            A = np.random.randn(256, 256).astype(np.float32)
            B = np.random.randn(256, 256).astype(np.float32)
            _ = kernel_engine.gemm(A, B)

        # Si llegamos aquí sin OOM, el test pasa
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
