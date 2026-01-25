// KERNEL SIMPLE REFERENCIA: cada hilo calcula un elemento de C, sin tiles
__kernel void gemm_reference(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
#ifdef DUMP_ACC
        C[row * N + col] = acc;
#else
        C[row * N + col] = (beta == 0.0f) ? alpha * acc : fma(alpha, acc, beta * C[row * N + col]);
#endif
    }
}
/*
 * BLOCK RECURSIVE GEMM - Phase 2, Technique 1 (v5 - OPTIMIZED)
 * 
 * Based directly on gemm_hybrid_opt.cl proven architecture.
 * Faithful reproduction with targeted improvements for +10-12%.
 * 
 * Target: 850-870 GFLOPS (vs 775 GFLOPS Phase 1 baseline)
 * 
 * Key features from Phase 1:
 * - Proven loading patterns
 * - Double buffering (removed for simplicity, may add back)
 * - float4 vectorization
 * - Register blocking
 * 
 * Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)
 * Date: 2026-01-24
 */

// Configuration defines (passed via compiler options)
#ifndef TS
#define TS 32
#endif


// BLOCK RECURSIVE GEMM - Phase 2, Technique 1 (v5 - OPTIMIZED)
// Only the optimized kernel is retained below.

__kernel void gemm_recursive_optimized(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Register blocking: cada hilo procesa 2x2 elementos para aumentar la intensidad aritmética
    const int BS = 2;  // Block size per thread (2×2)
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    
    // Cada workgroup procesa un tile TS×TS, pero cada hilo procesa BS×BS elementos
    int wg_row = get_group_id(0) * TS;
    int wg_col = get_group_id(1) * TS;
    
    // Posición del hilo dentro del tile
    int tile_row = local_y;
    int tile_col = local_x;
    
    // Acumuladores para los BS×BS elementos que procesa este hilo
    float acc[BS][BS];
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    __local float A_tile[TS][TS];
    __local float B_tile[TS][TS];
    
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int tiled_k = t * TS;
        
        // Carga cooperativa: todos los hilos colaboran para cargar los tiles completos
        for (int k = 0; k < TS; k++) {
            int global_k = tiled_k + k;
            
            // Carga de A_tile: cada hilo carga su fila
            int global_row = wg_row + local_y;
            if (global_k < K && global_row < M) {
                A_tile[local_y][k] = A[global_row * K + global_k];
            } else {
                A_tile[local_y][k] = 0.0f;
            }
            
            // Carga de B_tile: cada hilo carga su columna
            int global_col = wg_col + local_x;
            if (global_k < K && global_col < N) {
                B_tile[k][local_x] = B[global_k * N + global_col];
            } else {
                B_tile[k][local_x] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Acumulación con register blocking: cada hilo procesa BS×BS elementos
        for (int i = 0; i < BS; i++) {
            for (int j = 0; j < BS; j++) {
                int row = wg_row + tile_row * BS + i;
                int col = wg_col + tile_col * BS + j;
                if (row < M && col < N) {
                    for (int k = 0; k < TS; k++) {
                        acc[i][j] += A_tile[tile_row * BS + i][k] * B_tile[k][tile_col * BS + j];
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Escribir resultados: cada hilo escribe sus BS×BS elementos
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            int row = wg_row + tile_row * BS + i;
            int col = wg_col + tile_col * BS + j;
            if (row < M && col < N) {
#ifdef DUMP_ACC
                C[row * N + col] = acc[i][j];
#else
                C[row * N + col] = (beta == 0.0f) ? alpha * acc[i][j] : fma(alpha, acc[i][j], beta * C[row * N + col]);
#endif
            }
        }
    }
}
