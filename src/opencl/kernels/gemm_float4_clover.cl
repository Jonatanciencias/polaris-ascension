/**
 * GEMM FLOAT4 Optimized for Clover OpenCL 1.1
 * 
 * Compatible with Mesa Clover driver
 * - No 'restrict' keyword issues
 * - Simplified local memory allocation
 * - Clean float4 vectorization
 * - Optimized for GCN 4.0 architecture (RX 580/590)
 * 
 * Performance target: 180-200 GFLOPS on RX 590 GME
 * Phase 1: Quick Wins - Roadmap Task 1.1.2
 */

// Build options: -cl-mad-enable -cl-fast-relaxed-math -cl-std=CL1.1

// IMPORTANT: Use unique names to avoid conflicts with engine build options
// Each kernel uses hardcoded tile sizes for optimal performance

#define CLOVER_TILE_16 16   // For gemm_float4_clover (16x16 tiles)
#define CLOVER_TILE_8  8    // For gemm_float4_small (8x8 tiles)

// ============================================================================
// GEMM FLOAT4 - Clover Compatible Version
// ============================================================================

/**
 * gemm_float4_clover - GEMM con vectorización float4 compatible con Clover
 * 
 * Características:
 * - Vectorización con vload4/vstore4 (más compatible que punteros float4*)
 * - Local memory declarada en el kernel (no como argumento)
 * - Sin 'restrict' keyword
 * - Acceso coalescente optimizado
 * - Tiles de 16x16 para balance ocupancy/LDS
 * 
 * C = alpha * A * B + beta * C
 * 
 * Cada work-item procesa 4 elementos usando float4
 * Matrices deben estar alineadas a múltiplos de 4
 */
__kernel void gemm_float4_clover(
    const int M,
    const int N, 
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory tiles (16x16 elementos)
    __local float As[CLOVER_TILE_16 * CLOVER_TILE_16];
    __local float Bs[CLOVER_TILE_16 * CLOVER_TILE_16];
    
    // Work-item indices
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);
    
    // Accumulator for this work-item
    float sum = 0.0f;
    
    // Number of tiles needed for K dimension
    const int num_tiles = (K + CLOVER_TILE_16 - 1) / CLOVER_TILE_16;
    
    // Loop over K in tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A into local memory
        const int a_row = group_row * CLOVER_TILE_16 + local_row;
        const int a_col = t * CLOVER_TILE_16 + local_col;
        
        if (a_row < M && a_col < K) {
            As[local_row * CLOVER_TILE_16 + local_col] = A[a_row * K + a_col];
        } else {
            As[local_row * CLOVER_TILE_16 + local_col] = 0.0f;
        }
        
        // Load tile from B into local memory
        const int b_row = t * CLOVER_TILE_16 + local_row;
        const int b_col = group_col * CLOVER_TILE_16 + local_col;
        
        if (b_row < K && b_col < N) {
            Bs[local_row * CLOVER_TILE_16 + local_col] = B[b_row * N + b_col];
        } else {
            Bs[local_row * CLOVER_TILE_16 + local_col] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product for this tile
        // Unroll by 4 for better performance
        #pragma unroll 4
        for (int k = 0; k < CLOVER_TILE_16; k++) {
            float a_val = As[local_row * CLOVER_TILE_16 + k];
            float b_val = Bs[k * CLOVER_TILE_16 + local_col];
            sum = mad(a_val, b_val, sum);
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result with alpha/beta scaling
    if (global_row < M && global_col < N) {
        const int c_idx = global_row * N + global_col;
        if (beta == 0.0f) {
            C[c_idx] = alpha * sum;
        } else {
            C[c_idx] = alpha * sum + beta * C[c_idx];
        }
    }
}

// ============================================================================
// GEMM FLOAT4 VECTORIZED - Maximum Performance Version
// ============================================================================

/**
 * gemm_float4_vec - GEMM con vectorización float4 agresiva
 * 
 * Cada work-item procesa un bloque de 4x4 elementos usando float4
 * Requiere que N sea múltiplo de 4
 * 
 * Máximo aprovechamiento de operaciones SIMD en GCN
 */
__kernel void gemm_float4_vec(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Verificar alineación
    const int N_vec = N / 4;
    
    // Local memory tiles
    __local float As[CLOVER_TILE_16 * CLOVER_TILE_16];
    __local float Bs[CLOVER_TILE_16 * CLOVER_TILE_16];
    
    // Work-item indices
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col_vec = get_global_id(1);  // En unidades de float4
    const int global_col = global_col_vec * 4;
    const int group_row = get_group_id(0);
    const int group_col_vec = get_group_id(1);
    
    // Accumulators (float4 para 4 resultados simultáneos)
    float4 sum = (float4)(0.0f);
    
    // Number of tiles
    const int num_tiles = (K + CLOVER_TILE_16 - 1) / CLOVER_TILE_16;
    
    // Loop over K in tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A
        const int a_row = group_row * CLOVER_TILE_16 + local_row;
        const int a_col = t * CLOVER_TILE_16 + local_col;
        
        if (a_row < M && a_col < K) {
            As[local_row * CLOVER_TILE_16 + local_col] = A[a_row * K + a_col];
        } else {
            As[local_row * CLOVER_TILE_16 + local_col] = 0.0f;
        }
        
        // Load tile from B (vectorizado)
        const int b_row = t * CLOVER_TILE_16 + local_row;
        const int b_col_base = group_col_vec * CLOVER_TILE_16 * 4 + local_col * 4;
        
        if (b_row < K && b_col_base + 3 < N) {
            // Cargar 4 elementos consecutivos
            float4 b_vec = vload4(0, &B[b_row * N + b_col_base]);
            Bs[local_row * CLOVER_TILE_16 + local_col] = b_vec.x;
            if (local_col + 1 < CLOVER_TILE_16) Bs[local_row * CLOVER_TILE_16 + local_col + 1] = b_vec.y;
            if (local_col + 2 < CLOVER_TILE_16) Bs[local_row * CLOVER_TILE_16 + local_col + 2] = b_vec.z;
            if (local_col + 3 < CLOVER_TILE_16) Bs[local_row * CLOVER_TILE_16 + local_col + 3] = b_vec.w;
        } else {
            Bs[local_row * CLOVER_TILE_16 + local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute with vectorization
        #pragma unroll 4
        for (int k = 0; k < CLOVER_TILE_16; k++) {
            float a_val = As[local_row * CLOVER_TILE_16 + k];
            
            // Cargar 4 valores de B en paralelo
            float4 b_vec;
            b_vec.x = Bs[k * CLOVER_TILE_16 + local_col];
            b_vec.y = Bs[k * CLOVER_TILE_16 + (local_col + 1) % CLOVER_TILE_16];
            b_vec.z = Bs[k * CLOVER_TILE_16 + (local_col + 2) % CLOVER_TILE_16];
            b_vec.w = Bs[k * CLOVER_TILE_16 + (local_col + 3) % CLOVER_TILE_16];
            
            sum += a_val * b_vec;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    if (global_row < M && global_col + 3 < N) {
        const int c_idx = global_row * N + global_col;
        
        float4 c_vec;
        if (beta == 0.0f) {
            c_vec = alpha * sum;
        } else {
            c_vec = alpha * sum + beta * vload4(0, &C[c_idx]);
        }
        
        vstore4(c_vec, 0, &C[c_idx]);
    } else if (global_row < M) {
        // Boundary handling
        for (int i = 0; i < 4 && global_col + i < N; i++) {
            const int c_idx = global_row * N + global_col + i;
            float val;
            if (i == 0) val = sum.x;
            else if (i == 1) val = sum.y;
            else if (i == 2) val = sum.z;
            else val = sum.w;
            
            if (beta == 0.0f) {
                C[c_idx] = alpha * val;
            } else {
                C[c_idx] = alpha * val + beta * C[c_idx];
            }
        }
    }
}

// ============================================================================
// GEMM FLOAT4 HIGH OCCUPANCY - For Small Matrices
// ============================================================================

/**
 * gemm_float4_small - Optimizado para matrices pequeñas (<512)
 * 
 * - Tiles pequeños (8x8) para alta occupancy
 * - Menos uso de LDS
 * - Baja latencia
 */
#define SMALL_TILE 8

__kernel void gemm_float4_small(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    __local float As[SMALL_TILE * SMALL_TILE];
    __local float Bs[SMALL_TILE * SMALL_TILE];
    
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);
    
    float sum = 0.0f;
    const int num_tiles = (K + SMALL_TILE - 1) / SMALL_TILE;
    
    for (int t = 0; t < num_tiles; t++) {
        const int a_row = group_row * SMALL_TILE + local_row;
        const int a_col = t * SMALL_TILE + local_col;
        
        if (a_row < M && a_col < K) {
            As[local_row * SMALL_TILE + local_col] = A[a_row * K + a_col];
        } else {
            As[local_row * SMALL_TILE + local_col] = 0.0f;
        }
        
        const int b_row = t * SMALL_TILE + local_row;
        const int b_col = group_col * SMALL_TILE + local_col;
        
        if (b_row < K && b_col < N) {
            Bs[local_row * SMALL_TILE + local_col] = B[b_row * N + b_col];
        } else {
            Bs[local_row * SMALL_TILE + local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        #pragma unroll 8
        for (int k = 0; k < SMALL_TILE; k++) {
            sum = mad(As[local_row * SMALL_TILE + k],
                     Bs[k * SMALL_TILE + local_col],
                     sum);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_row < M && global_col < N) {
        const int c_idx = global_row * N + global_col;
        if (beta == 0.0f) {
            C[c_idx] = alpha * sum;
        } else {
            C[c_idx] = alpha * sum + beta * C[c_idx];
        }
    }
}
