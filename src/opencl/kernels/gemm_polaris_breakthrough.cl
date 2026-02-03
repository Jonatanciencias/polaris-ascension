/**
 * 🚀 POLARIS BREAKTHROUGH GEMM KERNEL
 * ====================================
 * Optimized for AMD Radeon RX 580/590 (Polaris 10/20)
 * 
 * Architecture optimizations:
 * - 36 Compute Units targeting
 * - 64KB LDS utilization
 * - 256-bit GDDR5 memory bus optimization
 * - GCN 4.0 wavefront (64 threads)
 * - Vectorized operations (float4/float8)
 * 
 * Author: Radeon RX 580 Framework
 * License: MIT
 */

// ============================================================================
// CONFIGURATION FOR POLARIS ARCHITECTURE
// ============================================================================

#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 16
#endif

#ifndef WORK_PER_THREAD_M
#define WORK_PER_THREAD_M 4
#endif

#ifndef WORK_PER_THREAD_N
#define WORK_PER_THREAD_N 4
#endif

// ============================================================================
// BASIC TILED GEMM - RELIABLE AND COMPATIBLE WITH MESA CLOVER
// ============================================================================

__kernel void gemm_basic_tiled(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C)
{
    // Local work-item IDs
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    // Work-group ID
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);
    
    // Global row and column
    const int global_row = group_row * TILE_M + local_row;
    const int global_col = group_col * TILE_N + local_col;
    
    // Local memory for tiles
    __local float A_tile[TILE_M][TILE_K];
    __local float B_tile[TILE_K][TILE_N];
    
    // Accumulator
    float sum = 0.0f;
    
    // Number of tiles
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Loop over tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load A tile
        const int a_row = global_row;
        const int a_col = t * TILE_K + local_col;
        if (a_row < M && a_col < K) {
            A_tile[local_row][local_col] = A[a_row * K + a_col];
        } else {
            A_tile[local_row][local_col] = 0.0f;
        }
        
        // Load B tile
        const int b_row = t * TILE_K + local_row;
        const int b_col = global_col;
        if (b_row < K && b_col < N) {
            B_tile[local_row][local_col] = B[b_row * N + b_col];
        } else {
            B_tile[local_row][local_col] = 0.0f;
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Multiply tiles
        for (int k = 0; k < TILE_K; k++) {
            sum += A_tile[local_row][k] * B_tile[k][local_col];
        }
        
        // Synchronize before next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = alpha * sum + beta * C[global_row * N + global_col];
    }
}

// ============================================================================
// OPTIMIZED GEMM WITH VECTORIZATION (float4)
// ============================================================================

__kernel void gemm_polaris_optimized(
    const int M, const int N, const int K,
    const float alpha,
    __global const float4* A,
    __global const float4* B,
    const float beta,
    __global float4* C)
{
    // Local IDs
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    // Global IDs
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);
    
    // Local memory
    __local float4 A_tile[TILE_M][TILE_K/4];
    __local float4 B_tile[TILE_K][TILE_N/4];
    
    // Accumulators for 4 elements
    float4 acc = (float4)(0.0f);
    
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Load A tile (vectorized)
        int a_idx = global_row * (K/4) + t * (TILE_K/4) + local_col;
        A_tile[local_row][local_col] = (global_row < M && (t * TILE_K + local_col * 4) < K) 
            ? A[a_idx] : (float4)(0.0f);
        
        // Load B tile (vectorized)
        int b_idx = (t * TILE_K + local_row) * (N/4) + global_col;
        B_tile[local_row][local_col] = ((t * TILE_K + local_row) < K && global_col * 4 < N) 
            ? B[b_idx] : (float4)(0.0f);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial products
        for (int k = 0; k < TILE_K; k++) {
            float4 a_val = A_tile[local_row][k/4];
            float4 b_val = B_tile[k][local_col];
            
            // Extract scalar from A based on k position
            float a_scalar = (k % 4 == 0) ? a_val.x :
                            (k % 4 == 1) ? a_val.y :
                            (k % 4 == 2) ? a_val.z : a_val.w;
            
            acc += a_scalar * b_val;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (global_row < M && global_col * 4 < N) {
        int c_idx = global_row * (N/4) + global_col;
        C[c_idx] = alpha * acc + beta * C[c_idx];
    }
}

// ============================================================================
// SIMPLE NAIVE GEMM - FOR CORRECTNESS VERIFICATION
// ============================================================================

__kernel void gemm_naive(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// BATCHED GEMM FOR MULTIPLE SMALL MATRICES
// ============================================================================

__kernel void gemm_batched(
    const int M, const int N, const int K,
    const int batch_size,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C)
{
    const int batch = get_global_id(2);
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (batch < batch_size && row < M && col < N) {
        const int stride_A = M * K;
        const int stride_B = K * N;
        const int stride_C = M * N;
        
        __global const float* A_batch = A + batch * stride_A;
        __global const float* B_batch = B + batch * stride_B;
        __global float* C_batch = C + batch * stride_C;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A_batch[row * K + k] * B_batch[k * N + col];
        }
        C_batch[row * N + col] = alpha * sum + beta * C_batch[row * N + col];
    }
}
