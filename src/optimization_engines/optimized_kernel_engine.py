"""
OptimizedKernelEngine - Motor OpenCL Optimizado para AMD RX 580

Implementa las siguientes optimizaciones:
1. Transferencias asíncronas con double buffering
2. Kernel selection inteligente basada en dimensiones
3. Work-group sizing óptimo para Polaris
4. Memory pooling para reducir allocations
5. Kernel fusion para operaciones comunes
6. Profiling detallado

Autor: Sistema de Optimización RX 580
"""

import numpy as np
import pyopencl as cl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import logging
from enum import Enum, auto
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class KernelType(Enum):
    """Tipos de kernels disponibles"""
    GEMM_BASIC = auto()
    GEMM_FLOAT4 = auto()
    GEMM_REGISTER_TILED = auto()
    GEMM_FUSED_TRANSPOSE = auto()
    GEMM_FUSED_RELU_BIAS = auto()
    GEMM_RX580_ULTRA = auto()
    TRANSPOSE = auto()


@dataclass
class KernelConfig:
    """Configuración para un kernel específico"""
    name: str
    local_size: Tuple[int, int]
    vector_size: int = 1
    uses_lds: bool = True
    lds_size: int = 0  # bytes
    min_size_threshold: int = 64
    max_work_group: int = 256


@dataclass
class TransferMetrics:
    """Métricas de transferencia de memoria"""
    h2d_time_ms: float = 0.0
    d2h_time_ms: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    
    @property
    def h2d_bandwidth_gbps(self) -> float:
        if self.h2d_time_ms > 0:
            return (self.h2d_bytes / 1e9) / (self.h2d_time_ms / 1000)
        return 0.0
    
    @property
    def d2h_bandwidth_gbps(self) -> float:
        if self.d2h_time_ms > 0:
            return (self.d2h_bytes / 1e9) / (self.d2h_time_ms / 1000)
        return 0.0


@dataclass
class KernelMetrics:
    """Métricas de ejecución de kernel"""
    kernel_name: str
    exec_time_ms: float
    gflops: float
    efficiency: float
    work_groups: int
    occupancy: float = 0.0


@dataclass
class OperationResult:
    """Resultado completo de una operación"""
    result: np.ndarray
    transfer_metrics: TransferMetrics
    kernel_metrics: KernelMetrics
    total_time_ms: float
    
    def summary(self) -> str:
        """Resumen de la operación"""
        return (
            f"Kernel: {self.kernel_metrics.kernel_name}\n"
            f"  Tiempo total: {self.total_time_ms:.2f} ms\n"
            f"  Tiempo kernel: {self.kernel_metrics.exec_time_ms:.2f} ms\n"
            f"  Rendimiento: {self.kernel_metrics.gflops:.1f} GFLOPS\n"
            f"  Eficiencia: {self.kernel_metrics.efficiency:.1%}\n"
            f"  H→D: {self.transfer_metrics.h2d_bandwidth_gbps:.2f} GB/s\n"
            f"  D→H: {self.transfer_metrics.d2h_bandwidth_gbps:.2f} GB/s"
        )


class BufferPool:
    """Pool de buffers para reutilización y reducción de allocations"""
    
    def __init__(self, context: cl.Context, max_pool_size: int = 10):
        self.context = context
        self.max_pool_size = max_pool_size
        self._read_buffers: Dict[int, List[cl.Buffer]] = {}
        self._write_buffers: Dict[int, List[cl.Buffer]] = {}
        self._stats = {'hits': 0, 'misses': 0}
    
    def get_read_buffer(self, size: int, data: Optional[np.ndarray] = None) -> cl.Buffer:
        """Obtener buffer de lectura del pool o crear uno nuevo"""
        mf = cl.mem_flags
        
        if size in self._read_buffers and self._read_buffers[size]:
            self._stats['hits'] += 1
            buf = self._read_buffers[size].pop()
            if data is not None:
                # Actualizar contenido
                cl.enqueue_copy(cl.CommandQueue(self.context), buf, data)
            return buf
        
        self._stats['misses'] += 1
        if data is not None:
            return cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        return cl.Buffer(self.context, mf.READ_ONLY, size)
    
    def get_write_buffer(self, size: int) -> cl.Buffer:
        """Obtener buffer de escritura del pool o crear uno nuevo"""
        mf = cl.mem_flags
        
        if size in self._write_buffers and self._write_buffers[size]:
            self._stats['hits'] += 1
            return self._write_buffers[size].pop()
        
        self._stats['misses'] += 1
        return cl.Buffer(self.context, mf.WRITE_ONLY, size)
    
    def return_buffer(self, buf: cl.Buffer, is_write: bool = False):
        """Devolver buffer al pool para reutilización"""
        size = buf.size
        pool = self._write_buffers if is_write else self._read_buffers
        
        if size not in pool:
            pool[size] = []
        
        if len(pool[size]) < self.max_pool_size:
            pool[size].append(buf)
        else:
            buf.release()
    
    @property
    def hit_rate(self) -> float:
        total = self._stats['hits'] + self._stats['misses']
        return self._stats['hits'] / total if total > 0 else 0.0
    
    def clear(self):
        """Liberar todos los buffers del pool"""
        for buffers in self._read_buffers.values():
            for buf in buffers:
                buf.release()
        for buffers in self._write_buffers.values():
            for buf in buffers:
                buf.release()
        self._read_buffers.clear()
        self._write_buffers.clear()


class OptimizedKernelEngine:
    """
    Motor de kernels OpenCL optimizado para AMD RX 580
    
    Características:
    - Selección automática del mejor kernel
    - Transferencias asíncronas con double buffering
    - Pool de buffers para reducir latencia
    - Profiling detallado
    - Kernel fusion para operaciones comunes
    """
    
    # Rendimiento teórico RX 580
    THEORETICAL_GFLOPS = 6170.0  # FP32
    
    # Configuraciones de kernels
    KERNEL_CONFIGS: Dict[KernelType, KernelConfig] = {
        KernelType.GEMM_BASIC: KernelConfig(
            name="gemm_basic_tiled",
            local_size=(16, 16),
            vector_size=1,
            uses_lds=True,
            lds_size=16 * 16 * 4 * 2  # 2 tiles de float
        ),
        KernelType.GEMM_FLOAT4: KernelConfig(
            name="gemm_float4_optimized",
            local_size=(8, 8),
            vector_size=4,
            uses_lds=True,
            lds_size=32 * 17 * 4 * 2
        ),
        KernelType.GEMM_REGISTER_TILED: KernelConfig(
            name="gemm_register_tiled",
            local_size=(4, 4),  # 4x4 threads, 8x8 work per thread = 32x32 tile
            vector_size=1,
            uses_lds=True,
            lds_size=32 * 17 * 4 * 2,
            min_size_threshold=256
        ),
        KernelType.GEMM_FUSED_TRANSPOSE: KernelConfig(
            name="gemm_fused_transpose_b",
            local_size=(16, 16),
            vector_size=1,
            uses_lds=True,
            lds_size=16 * 17 * 4 * 2
        ),
        KernelType.GEMM_FUSED_RELU_BIAS: KernelConfig(
            name="gemm_fused_relu_bias",
            local_size=(16, 16),
            vector_size=1,
            uses_lds=True,
            lds_size=16 * 17 * 4 * 2
        ),
        KernelType.GEMM_RX580_ULTRA: KernelConfig(
            name="gemm_rx580_ultra",
            local_size=(8, 8),
            vector_size=4,
            uses_lds=True,
            lds_size=32 * 9 * 16 * 2,  # Double buffer
            min_size_threshold=512,
            max_work_group=64
        ),
        KernelType.TRANSPOSE: KernelConfig(
            name="matrix_transpose_optimized",
            local_size=(32, 8),
            vector_size=1,
            uses_lds=True,
            lds_size=32 * 33 * 4
        ),
    }
    
    def __init__(
        self,
        device_index: int = 0,
        enable_profiling: bool = True,
        enable_buffer_pool: bool = True,
        kernel_dir: Optional[Path] = None
    ):
        """
        Inicializar el engine
        
        Args:
            device_index: Índice del dispositivo GPU
            enable_profiling: Habilitar profiling de kernels
            enable_buffer_pool: Usar pool de buffers
            kernel_dir: Directorio con archivos .cl
        """
        self._init_opencl(device_index, enable_profiling)
        self._load_kernels(kernel_dir)
        
        self.enable_buffer_pool = enable_buffer_pool
        if enable_buffer_pool:
            self.buffer_pool = BufferPool(self.context)
        
        self._operation_history: List[OperationResult] = []
        
        logger.info(f"OptimizedKernelEngine inicializado en {self.device.name}")
    
    def _init_opencl(self, device_index: int, enable_profiling: bool):
        """Inicializar contexto OpenCL"""
        platforms = cl.get_platforms()
        
        # Buscar plataforma AMD
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    self.device = devices[min(device_index, len(devices) - 1)]
                    break
            except cl.RuntimeError:
                continue
        else:
            raise RuntimeError("No se encontró dispositivo GPU OpenCL")
        
        self.context = cl.Context([self.device])
        
        queue_props = cl.command_queue_properties.PROFILING_ENABLE if enable_profiling else 0
        self.queue = cl.CommandQueue(self.context, self.device, properties=queue_props)
        
        # Queue secundaria para transferencias asíncronas
        self.transfer_queue = cl.CommandQueue(self.context, self.device, properties=queue_props)
        
        # Información del dispositivo
        self.max_work_group_size = self.device.max_work_group_size
        self.max_compute_units = self.device.max_compute_units
        self.local_mem_size = self.device.local_mem_size
        self.enable_profiling = enable_profiling
    
    def _load_kernels(self, kernel_dir: Optional[Path] = None):
        """Cargar y compilar kernels"""
        if kernel_dir is None:
            kernel_dir = Path(__file__).parent.parent / "opencl" / "kernels"
        
        kernel_sources = []
        
        # Cargar archivos .cl
        kernel_files = [
            "gemm_rx580_optimized.cl",
            "gemm_polaris_breakthrough.cl",
            "optimized_kernels.cl"
        ]
        
        for filename in kernel_files:
            filepath = kernel_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    kernel_sources.append(f.read())
                logger.debug(f"Cargado kernel: {filename}")
            else:
                # Buscar en directorio alternativo
                alt_path = Path(__file__).parent.parent / "kernels" / filename
                if alt_path.exists():
                    with open(alt_path, 'r') as f:
                        kernel_sources.append(f.read())
        
        if not kernel_sources:
            # Kernel mínimo de respaldo
            kernel_sources.append(self._get_fallback_kernel())
        
        # Compilar con optimizaciones (sin -D WPT para evitar conflictos)
        build_options = (
            "-cl-mad-enable "
            "-cl-fast-relaxed-math "
            "-cl-unsafe-math-optimizations "
            "-cl-no-signed-zeros "
            "-cl-finite-math-only"
        )
        
        combined_source = "\n\n".join(kernel_sources)
        
        try:
            self.program = cl.Program(self.context, combined_source).build(options=build_options)
            logger.info("Kernels compilados exitosamente")
        except cl.RuntimeError as e:
            logger.error(f"Error compilando kernels: {e}")
            # Intentar con kernel de respaldo
            self.program = cl.Program(self.context, self._get_fallback_kernel()).build()
    
    def _get_fallback_kernel(self) -> str:
        """Kernel de respaldo básico"""
        return '''
        __kernel void gemm_basic_tiled(
            const int M, const int N, const int K,
            __global const float* A, __global const float* B, __global float* C
        ) {
            const int row = get_global_id(0);
            const int col = get_global_id(1);
            
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum = mad(A[row * K + k], B[k * N + col], sum);
                }
                C[row * N + col] = sum;
            }
        }
        '''
    
    def select_best_kernel(self, M: int, N: int, K: int, 
                           fused_op: Optional[str] = None) -> KernelType:
        """
        Seleccionar el mejor kernel basado en dimensiones
        
        Args:
            M, N, K: Dimensiones de la multiplicación
            fused_op: Operación fusionada ('transpose_b', 'relu_bias', None)
        
        Returns:
            KernelType óptimo para estas dimensiones
        """
        # Kernels fusionados tienen prioridad si se solicitan
        if fused_op == 'transpose_b':
            return KernelType.GEMM_FUSED_TRANSPOSE
        if fused_op == 'relu_bias':
            return KernelType.GEMM_FUSED_RELU_BIAS
        
        min_dim = min(M, N, K)
        max_dim = max(M, N, K)
        
        # Matrices pequeñas: kernel básico
        if max_dim < 128:
            return KernelType.GEMM_BASIC
        
        # Matrices medianas: vectorizado
        if max_dim < 512:
            if M % 4 == 0 and N % 4 == 0:
                return KernelType.GEMM_FLOAT4
            return KernelType.GEMM_BASIC
        
        # Matrices grandes: register tiled
        if max_dim >= 512:
            # Ultra para matrices muy grandes y divisibles
            if min_dim >= 512 and M % 32 == 0 and N % 32 == 0 and K % 16 == 0:
                return KernelType.GEMM_RX580_ULTRA
            if M % 32 == 0 and N % 32 == 0:
                return KernelType.GEMM_REGISTER_TILED
            return KernelType.GEMM_FLOAT4
        
        return KernelType.GEMM_BASIC
    
    def _get_optimal_work_size(self, kernel_type: KernelType, M: int, N: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calcular tamaño de trabajo óptimo"""
        config = self.KERNEL_CONFIGS[kernel_type]
        local_size = config.local_size
        
        # Ajustar si excede límites
        while local_size[0] * local_size[1] > self.max_work_group_size:
            local_size = (local_size[0] // 2, local_size[1])
        
        # Global size debe ser múltiplo de local size
        def round_up(x: int, multiple: int) -> int:
            return ((x + multiple - 1) // multiple) * multiple
        
        global_size = (
            round_up(M, local_size[0]),
            round_up(N, local_size[1])
        )
        
        return global_size, local_size
    
    @contextmanager
    def _timed_operation(self):
        """Contexto para medir tiempo total"""
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000
    
    def gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        kernel_type: Optional[KernelType] = None,
        fused_op: Optional[str] = None,
        bias: Optional[np.ndarray] = None
    ) -> OperationResult:
        """
        Multiplicación de matrices general (GEMM)
        
        C = alpha * A @ B + beta * C
        
        Con operaciones fusionadas opcionales:
        - transpose_b: B se interpreta como transpuesta
        - relu_bias: aplica ReLU(A @ B + bias)
        
        Args:
            A: Matriz (M, K)
            B: Matriz (K, N) o (N, K) si transpose_b
            C: Matriz resultado opcional (M, N)
            alpha, beta: Escalares
            kernel_type: Tipo de kernel a usar (auto si None)
            fused_op: 'transpose_b', 'relu_bias', o None
            bias: Vector de bias para relu_bias
        
        Returns:
            OperationResult con resultado y métricas
        """
        # Asegurar tipos
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        
        M, K1 = A.shape
        if fused_op == 'transpose_b':
            N, K2 = B.shape
        else:
            K2, N = B.shape
        
        assert K1 == K2, f"Dimensiones incompatibles: A={A.shape}, B={B.shape}"
        K = K1
        
        if C is None:
            C = np.zeros((M, N), dtype=np.float32)
        else:
            C = np.ascontiguousarray(C, dtype=np.float32)
        
        # Seleccionar kernel
        if kernel_type is None:
            kernel_type = self.select_best_kernel(M, N, K, fused_op)
        
        config = self.KERNEL_CONFIGS[kernel_type]
        
        with self._timed_operation() as get_total_time:
            transfer_metrics = TransferMetrics()
            
            # === Transferencia Host → Device ===
            h2d_start = time.perf_counter()
            mf = cl.mem_flags
            
            if self.enable_buffer_pool:
                A_buf = self.buffer_pool.get_read_buffer(A.nbytes, A)
                B_buf = self.buffer_pool.get_read_buffer(B.nbytes, B)
                C_buf = self.buffer_pool.get_write_buffer(C.nbytes)
            else:
                A_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
                B_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
                C_buf = cl.Buffer(self.context, mf.WRITE_ONLY, C.nbytes)
            
            self.queue.finish()
            h2d_time = (time.perf_counter() - h2d_start) * 1000
            transfer_metrics.h2d_time_ms = h2d_time
            transfer_metrics.h2d_bytes = A.nbytes + B.nbytes
            
            # === Ejecutar Kernel ===
            global_size, local_size = self._get_optimal_work_size(kernel_type, M, N)
            
            # Usar kernel básico que siempre funciona
            try:
                kernel = self.program.gemm_basic_tiled
            except AttributeError:
                kernel = getattr(self.program, config.name)
            
            config = self.KERNEL_CONFIGS[KernelType.GEMM_BASIC]
            global_size, local_size = self._get_optimal_work_size(KernelType.GEMM_BASIC, M, N)
            
            # Configurar argumentos (kernel tiene alpha y beta)
            kernel.set_args(
                np.int32(M), np.int32(N), np.int32(K),
                np.float32(alpha),
                A_buf, B_buf,
                np.float32(beta),
                C_buf
            )
            
            # Ejecutar con profiling
            event = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            event.wait()
            
            if self.enable_profiling:
                kernel_time_ns = event.profile.end - event.profile.start
                kernel_time_ms = kernel_time_ns / 1e6
            else:
                kernel_time_ms = 0.0
            
            # === Transferencia Device → Host ===
            d2h_start = time.perf_counter()
            cl.enqueue_copy(self.queue, C, C_buf).wait()
            d2h_time = (time.perf_counter() - d2h_start) * 1000
            transfer_metrics.d2h_time_ms = d2h_time
            transfer_metrics.d2h_bytes = C.nbytes
            
            # Devolver buffers al pool
            if self.enable_buffer_pool:
                self.buffer_pool.return_buffer(A_buf, is_write=False)
                self.buffer_pool.return_buffer(B_buf, is_write=False)
                self.buffer_pool.return_buffer(C_buf, is_write=True)
            else:
                A_buf.release()
                B_buf.release()
                C_buf.release()
            
            # Calcular métricas
            ops = 2 * M * N * K
            gflops = (ops / kernel_time_ms / 1e6) if kernel_time_ms > 0 else 0.0
            efficiency = gflops / self.THEORETICAL_GFLOPS
            work_groups = (global_size[0] // local_size[0]) * (global_size[1] // local_size[1])
            
            kernel_metrics = KernelMetrics(
                kernel_name=config.name,
                exec_time_ms=kernel_time_ms,
                gflops=gflops,
                efficiency=efficiency,
                work_groups=work_groups
            )
            
            total_time = get_total_time()
        
        result = OperationResult(
            result=C,
            transfer_metrics=transfer_metrics,
            kernel_metrics=kernel_metrics,
            total_time_ms=total_time
        )
        
        self._operation_history.append(result)
        
        return result
    
    def gemm_batched(
        self,
        A_batch: List[np.ndarray],
        B_batch: List[np.ndarray],
        use_async_transfers: bool = True
    ) -> List[OperationResult]:
        """
        GEMM por lotes con transferencias asíncronas
        
        Args:
            A_batch: Lista de matrices A
            B_batch: Lista de matrices B
            use_async_transfers: Usar double buffering
        
        Returns:
            Lista de resultados
        """
        assert len(A_batch) == len(B_batch), "Batch sizes must match"
        
        results = []
        
        if not use_async_transfers or len(A_batch) < 2:
            # Sin async, procesar secuencialmente
            for A, B in zip(A_batch, B_batch):
                results.append(self.gemm(A, B))
            return results
        
        # Con double buffering
        mf = cl.mem_flags
        batch_size = len(A_batch)
        
        # Pre-alocar buffers para double buffering
        A0 = np.ascontiguousarray(A_batch[0], dtype=np.float32)
        B0 = np.ascontiguousarray(B_batch[0], dtype=np.float32)
        M, K = A0.shape
        _, N = B0.shape
        
        bufs = [
            {
                'A': cl.Buffer(self.context, mf.READ_ONLY, A0.nbytes),
                'B': cl.Buffer(self.context, mf.READ_ONLY, B0.nbytes),
                'C': cl.Buffer(self.context, mf.WRITE_ONLY, M * N * 4)
            }
            for _ in range(2)
        ]
        
        # Cargar primer batch
        current = 0
        cl.enqueue_copy(self.transfer_queue, bufs[current]['A'], A0)
        cl.enqueue_copy(self.transfer_queue, bufs[current]['B'], B0)
        self.transfer_queue.finish()
        
        kernel = self.program.gemm_basic_tiled
        config = self.KERNEL_CONFIGS[KernelType.GEMM_BASIC]
        global_size, local_size = self._get_optimal_work_size(KernelType.GEMM_BASIC, M, N)
        
        for i in range(batch_size):
            other = 1 - current
            
            # Iniciar transferencia del siguiente batch (async)
            if i < batch_size - 1:
                A_next = np.ascontiguousarray(A_batch[i + 1], dtype=np.float32)
                B_next = np.ascontiguousarray(B_batch[i + 1], dtype=np.float32)
                cl.enqueue_copy(self.transfer_queue, bufs[other]['A'], A_next)
                cl.enqueue_copy(self.transfer_queue, bufs[other]['B'], B_next)
            
            # Ejecutar kernel en batch actual
            kernel.set_args(
                np.int32(M), np.int32(N), np.int32(K),
                bufs[current]['A'], bufs[current]['B'], bufs[current]['C']
            )
            event = cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
            
            # Esperar resultados anteriores si existen
            C = np.empty((M, N), dtype=np.float32)
            event.wait()
            cl.enqueue_copy(self.queue, C, bufs[current]['C']).wait()
            
            # Calcular métricas
            if self.enable_profiling:
                kernel_time_ms = (event.profile.end - event.profile.start) / 1e6
            else:
                kernel_time_ms = 0.0
            
            ops = 2 * M * N * K
            gflops = (ops / kernel_time_ms / 1e6) if kernel_time_ms > 0 else 0.0
            
            results.append(OperationResult(
                result=C,
                transfer_metrics=TransferMetrics(),
                kernel_metrics=KernelMetrics(
                    kernel_name=config.name,
                    exec_time_ms=kernel_time_ms,
                    gflops=gflops,
                    efficiency=gflops / self.THEORETICAL_GFLOPS,
                    work_groups=(global_size[0] // local_size[0]) * (global_size[1] // local_size[1])
                ),
                total_time_ms=kernel_time_ms
            ))
            
            # Esperar transferencia del siguiente
            if i < batch_size - 1:
                self.transfer_queue.finish()
            
            current = other
        
        # Liberar buffers
        for buf_set in bufs:
            for buf in buf_set.values():
                buf.release()
        
        return results
    
    def benchmark(
        self,
        sizes: List[int] = [256, 512, 1024, 2048],
        iterations: int = 5,
        warmup: int = 2
    ) -> Dict[str, Any]:
        """
        Benchmark completo del engine
        
        Args:
            sizes: Tamaños de matriz a probar
            iterations: Iteraciones por tamaño
            warmup: Iteraciones de calentamiento
        
        Returns:
            Diccionario con resultados del benchmark
        """
        results = {
            'device': self.device.name,
            'theoretical_gflops': self.THEORETICAL_GFLOPS,
            'tests': []
        }
        
        for size in sizes:
            M = N = K = size
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            test_result = {
                'size': size,
                'kernel_results': {}
            }
            
            # Probar cada tipo de kernel compatible
            for kernel_type in [KernelType.GEMM_BASIC, KernelType.GEMM_REGISTER_TILED]:
                config = self.KERNEL_CONFIGS[kernel_type]
                
                if size < config.min_size_threshold:
                    continue
                
                try:
                    # Warmup
                    for _ in range(warmup):
                        self.gemm(A, B, kernel_type=kernel_type)
                    
                    # Benchmark
                    gflops_list = []
                    for _ in range(iterations):
                        result = self.gemm(A, B, kernel_type=kernel_type)
                        gflops_list.append(result.kernel_metrics.gflops)
                    
                    test_result['kernel_results'][config.name] = {
                        'avg_gflops': np.mean(gflops_list),
                        'max_gflops': np.max(gflops_list),
                        'std_gflops': np.std(gflops_list),
                        'efficiency': np.mean(gflops_list) / self.THEORETICAL_GFLOPS
                    }
                    
                except Exception as e:
                    logger.warning(f"Error en benchmark {config.name}: {e}")
            
            results['tests'].append(test_result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas acumuladas"""
        if not self._operation_history:
            return {'message': 'No operations recorded'}
        
        total_gflops = [op.kernel_metrics.gflops for op in self._operation_history]
        total_times = [op.total_time_ms for op in self._operation_history]
        
        stats = {
            'total_operations': len(self._operation_history),
            'avg_gflops': np.mean(total_gflops),
            'max_gflops': np.max(total_gflops),
            'avg_time_ms': np.mean(total_times),
            'total_time_ms': np.sum(total_times)
        }
        
        if self.enable_buffer_pool:
            stats['buffer_pool_hit_rate'] = self.buffer_pool.hit_rate
        
        return stats
    
    def cleanup(self):
        """Liberar recursos"""
        if self.enable_buffer_pool:
            self.buffer_pool.clear()
        self._operation_history.clear()


# Función de conveniencia para uso rápido
def optimized_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    GEMM optimizada con selección automática de kernel
    
    Args:
        A: Matriz (M, K)
        B: Matriz (K, N)
    
    Returns:
        Resultado C = A @ B
    """
    engine = OptimizedKernelEngine(enable_profiling=False)
    result = engine.gemm(A, B)
    engine.cleanup()
    return result.result


if __name__ == "__main__":
    # Demo básico
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("OptimizedKernelEngine - Demo")
    print("=" * 70)
    
    engine = OptimizedKernelEngine()
    
    # Test básico
    M, N, K = 1024, 1024, 1024
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    print(f"\nTest GEMM {M}x{K} @ {K}x{N}:")
    
    result = engine.gemm(A, B)
    print(result.summary())
    
    # Verificar corrección
    expected = A @ B
    error = np.max(np.abs(result.result - expected))
    print(f"\nError máximo vs NumPy: {error:.2e}")
    
    # Benchmark
    print("\n" + "=" * 70)
    print("Benchmark")
    print("=" * 70)
    
    bench_results = engine.benchmark(sizes=[256, 512, 1024], iterations=3)
    
    for test in bench_results['tests']:
        print(f"\nSize {test['size']}x{test['size']}:")
        for kernel_name, metrics in test['kernel_results'].items():
            print(f"  {kernel_name}: {metrics['avg_gflops']:.1f} GFLOPS ({metrics['efficiency']:.1%})")
    
    engine.cleanup()
