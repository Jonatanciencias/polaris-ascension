import numpy as np
from opencl.gemm_recursive_wrapper import RecursiveGEMMExecutor, RecursiveConfig

np.random.seed(0)
A = np.random.randn(64, 64).astype(np.float32)
B = np.random.randn(64, 64).astype(np.float32)

executor = RecursiveGEMMExecutor(RecursiveConfig(kernel_variant="optimized"))
C_gpu = executor.gemm(A, B)
C_np = A @ B

np.set_printoptions(precision=4, suppress=True)
print("Diferencia absoluta máxima:", np.max(np.abs(C_gpu - C_np)))
print("Diferencia absoluta media:", np.mean(np.abs(C_gpu - C_np)))
