import numpy as np
from opencl.gemm_recursive_wrapper import RecursiveGEMMExecutor, RecursiveConfig

# Prueba pequeña: matrices 8x8
np.random.seed(0)
A = np.random.randn(8, 8).astype(np.float32)
B = np.random.randn(8, 8).astype(np.float32)

# Ejecutar en GPU
executor = RecursiveGEMMExecutor(RecursiveConfig(kernel_variant="optimized"))
C_gpu = executor.gemm(A, B)

# Ejecutar en NumPy
C_np = A @ B

# Mostrar ambas salidas y la diferencia
np.set_printoptions(precision=4, suppress=True)
print("C (GPU):\n", C_gpu)
print("C (NumPy):\n", C_np)
print("Diferencia absoluta máxima:", np.max(np.abs(C_gpu - C_np)))
print("Diferencia absoluta media:", np.mean(np.abs(C_gpu - C_np)))
