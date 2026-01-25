#!/usr/bin/env python3
"""
FASE 6: Winograd GEMM Proof of Concept
Script para probar el kernel Winograd W(2×2, 3×3)

Fecha: Enero 2026
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    print("PyOpenCL not available")
    sys.exit(1)

class WinogradGEMMPOC:
    """Proof of Concept para Winograd GEMM W(2×2, 3×3)"""

    def __init__(self):
        self.context = None
        self.queue = None
        self.program = None
        self.kernel = None

        self.initialize_opencl()

    def initialize_opencl(self):
        """Inicializar OpenCL context y queue"""
        try:
            # Seleccionar plataforma y device
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            # Usar primera plataforma (AMD)
            platform = platforms[0]
            print(f"Using platform: {platform.name}")

            # Seleccionar GPU
            devices = platform.get_devices(cl.device_type.GPU)
            if not devices:
                raise RuntimeError("No GPU devices found")

            device = devices[0]
            print(f"Using device: {device.name}")

            # Crear context y queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)

            print("OpenCL initialized successfully")

        except Exception as e:
            print(f"OpenCL initialization failed: {e}")
            sys.exit(1)

    def load_kernel(self):
        """Cargar y compilar el kernel Winograd"""
        kernel_path = Path(__file__).parent / 'gemm_winograd_w2x2.cl'

        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

        with open(kernel_path, 'r') as f:
            kernel_source = f.read()

        try:
            # Compilar programa
            self.program = cl.Program(self.context, kernel_source).build()

            # Obtener kernel
            self.kernel = self.program.gemm_winograd_w2x2_basic

            print("Kernel compiled successfully")

        except cl.BuildError as e:
            print(f"Kernel compilation failed: {e}")
            print("Build log:")
            print(e.build_log)
            sys.exit(1)
        except Exception as e:
            print(f"Kernel loading failed: {e}")
            sys.exit(1)

    def create_test_data(self):
        """Crear datos de prueba para validación"""
        # Para W(2×2, 3×3):
        # - Input tile: 4×4
        # - Kernel: 3×3
        # - Output: 2×2

        # Crear input tile 4×4 (dummy data)
        input_tile = np.eye(4, dtype=np.float32)  # Identity matrix for simpler debugging
        print("Input tile 4×4:")
        print(input_tile)

        # Crear kernel 3×3 (identidad para testing)
        kernel = np.eye(3, dtype=np.float32)
        print("\nKernel 3×3 (identidad):")
        print(kernel)

        return input_tile, kernel

    def reference_winograd(self, input_tile, kernel):
        """Implementación de referencia Winograd completa en NumPy"""
        # Transform matrices correctas
        G = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [-1, 1, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.float32)

        BT = np.array([
            [1, 0, 0],
            [0, 1, -1],
            [0, 1, 1],
            [0, 0, 0]
        ], dtype=np.float32)

        AT = np.array([
            [1, 1, 1, 0],
            [0, 1, -1, -1]
        ], dtype=np.float32)

        AT_T = np.array([
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1]
        ], dtype=np.float32)

        print("\nWinograd Transform Matrices:")
        print("G (4x4):")
        print(G)
        print("BT (4x3):")
        print(BT)
        print("AT (2x4):")
        print(AT)
        print("AT_T (4x2):")
        print(AT_T)

        # Paso 1: Input transform U = G * input_tile * G^T
        U = G @ input_tile @ G.T
        print("\nPaso 1 - Input transform U = G * input_tile * G^T:")
        print(U)

        # Paso 2: Kernel transform V = BT * kernel
        V = BT @ kernel
        # Pad to 4x4
        V_padded = np.zeros((4, 4), dtype=np.float32)
        V_padded[:, :3] = V
        V_padded[:, 3] = V_padded[:, 2]  # Copy last column
        print("\nPaso 2 - Kernel transform V = BT * kernel (padded to 4x4):")
        print(V_padded)

        # Paso 3: Element-wise multiplication M = U ⊙ V
        M = U * V_padded
        print("\nPaso 3 - Element-wise product M = U ⊙ V:")
        print(M)

        # Paso 4: Output transform C = AT * M * AT_T
        C = AT @ M @ AT_T
        print("\nPaso 4 - Output transform C = AT * M * AT_T:")
        print(C)

        return C

    def run_kernel_test(self, input_tile, kernel):
        """Ejecutar kernel OpenCL completo y obtener resultado"""
        # Dimensiones para POC
        M, N, K = 2, 2, 4  # Output 2×2, input dimension 4

        # Crear matrices dummy para A y B (no se usan en el kernel POC)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        # Crear buffers
        A_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, C.nbytes)

        # Configurar kernel arguments
        self.kernel.set_args(A_buf, B_buf, C_buf, np.int32(M), np.int32(N), np.int32(K))

        # Ejecutar kernel (solo un work item para POC)
        global_size = (1, 1)
        local_size = (1, 1)

        print("\nEjecutando kernel OpenCL completo...")
        start_time = time.time()
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, local_size)
        self.queue.finish()
        end_time = time.time()

        # Leer resultado
        cl.enqueue_copy(self.queue, C, C_buf)
        self.queue.finish()

        execution_time = end_time - start_time

        print(f"\nKernel execution time: {execution_time:.6f} seconds")
        print("GPU Output (2x2 result):")
        print(C)

        return C, execution_time

    def validate_results(self, gpu_result, cpu_result):
        """Validar que GPU y CPU dan resultados similares"""
        diff = np.abs(gpu_result - cpu_result)
        max_error = np.max(diff)
        mean_error = np.mean(diff)

        print("\nValidation Results:")
        print(f"Max error: {max_error}")
        print(f"Mean error: {mean_error}")

        if max_error < 1e-5:
            print("✅ VALIDATION PASSED")
            return True
        else:
            print("❌ VALIDATION FAILED")
            print("GPU result:")
            print(gpu_result)
            print("CPU result:")
            print(cpu_result)
            return False

def main():
    """Función principal del POC"""
    print("🚀 FASE 6: Winograd GEMM Proof of Concept")
    print("=" * 50)

    # Inicializar
    poc = WinogradGEMMPOC()
    poc.load_kernel()

    # Crear datos de prueba
    input_tile, kernel = poc.create_test_data()

    # Ejecutar referencia en CPU
    print("\n" + "=" * 50)
    print("REFERENCE IMPLEMENTATION (NumPy)")
    print("=" * 50)
    cpu_result = poc.reference_winograd(input_tile, kernel)

    # Ejecutar kernel GPU
    print("\n" + "=" * 50)
    print("GPU KERNEL EXECUTION")
    print("=" * 50)
    gpu_result, exec_time = poc.run_kernel_test(input_tile, kernel)

    # Validar resultados
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    success = poc.validate_results(gpu_result, cpu_result)

    # Resultados finales
    print("\n" + "=" * 50)
    print("POC RESULTS SUMMARY")
    print("=" * 50)
    print(f"Kernel execution time: {exec_time:.6f} seconds")
    print(f"Validation: {'PASSED' if success else 'FAILED'}")

    if success:
        print("🎉 Winograd POC successful!")
        print("Next steps: Implement real data loading and optimize performance")
    else:
        print("⚠️  Issues found - debug required before proceeding")

if __name__ == "__main__":
    main()