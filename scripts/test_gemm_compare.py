import numpy as np
from opencl.gemm_recursive_wrapper import RecursiveGEMMExecutor, RecursiveConfig

np.random.seed(0)
A = np.random.randn(64, 64).astype(np.float32)
B = np.random.randn(64, 64).astype(np.float32)

# Parámetros de test
TS = 8
row, col = 22, 10
print(f"\n[INFO] Parámetros: TS={TS}, row={row}, col={col}, shape A={A.shape}, shape B={B.shape}")
print(f"[INFO] local_y={row%TS}, local_x={col%TS}, group_y={row//TS}, group_x={col//TS}")

# Mostrar valores de entrada relevantes
print(f"\n[INFO] A[{row}, 0:{TS}]:", A[row, :TS])
print(f"[INFO] B[0:{TS}, {col}]:", B[:TS, col])


# Ejecutar kernel de referencia (sin tiles, dump_acc)
executor_ref = RecursiveGEMMExecutor(RecursiveConfig(kernel_variant="reference", dump_acc=True))
C_ref = executor_ref.gemm(A, B)

# Ejecutar kernel tiled (optimizado, dump_acc)
executor_opt = RecursiveGEMMExecutor(RecursiveConfig(kernel_variant="optimized", dump_acc=True))
C_opt = executor_opt.gemm(A, B)



np.set_printoptions(precision=4, suppress=True)

# Comparar solo la matriz real (sin filas de volcado extra)
M, N = A.shape[0], B.shape[1]
diff = np.abs(C_opt[:M, :N] - C_ref[:M, :N])
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print("Diferencia acumuladores (tiled vs referencia, max, mean):", max_diff, mean_diff)
if max_diff > 1e-4:
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Máxima diferencia en {idx}: tiled={C_opt[:M, :N][idx]}, ref={C_ref[:M, :N][idx]}, diff={diff[idx]}")


# Mostrar el volcado de tiles para la celda (22,10) en C_opt
print("\n[INFO] Fila 22 de C_opt (A_tile[0:8], B_tile[0:8] en columnas 0-15):")
print(C_opt[22, :16])

# Comparación directa de los valores esperados de A_tile y B_tile para la celda (22,10)
print("[INFO] Valores esperados para A_tile (A[22, 0:8]):", A[22, :8])
print("[INFO] Valores esperados para B_tile (B[0:8, 10]):", B[:8, 10])

# Si hay fila 23, mostrar volcado de índices/valores globales si el kernel lo produce
if C_opt.shape[0] > 23:
    print("[INFO] Fila 23 de C_opt (volcado de valores globales A[22,k] y B[k,10] si está implementado):")
    print(C_opt[23, :16])
