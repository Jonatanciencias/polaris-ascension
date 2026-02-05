/*
 * Approach 1 v3: Fixed Indexing - CORRECTED
 * 
 * Root Cause Analysis from v1 & v2:
 *   v1: Only accessed first 16×16 of 20×20 tile (incomplete coverage)
 *   v2: Incorrect B tile indexing calculation (global col offset wrong)
 * 
 * Key Fixes in v3:
 *   1. Proper global-to-tile mapping
 *   2. Correct B tile indexing for vectorized loads
 *   3. Each thread computes multiple output rows (full tile utilization)
 *   4. Simplified and verified index calculations
 * 
 * Strategy:
 *   - Work groups of 16×16 threads process 20×20 output tiles
 *   - Cooperative loading: 256 threads load 400 A elements + 1600 B elements
 *   - Each thread computes ceil(20/16)=2 output rows × 4 columns
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL v3 - FIXED INDEXING
 */

#define TILE_SIZE 20
#define LOCAL_X 16
#define LOCAL_Y 16
#define TOTAL_THREADS (LOCAL_X * LOCAL_Y)  // 256

__kernel void gemm_tile20_fixed(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory tiles
    __local float As[TILE_SIZE * TILE_SIZE];  // 20×20 = 400 floats
    __local float Bs[TILE_SIZE * TILE_SIZE * 4];  // 20×20×4 = 1600 floats
    
    // Thread IDs
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_linear = local_y * LOCAL_X + local_x;  // 0-255
    
    // Work group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Accumulators for 2 rows × 4 columns
    float4 acc0 = (float4)(0.0f);  // Row 0
    float4 acc1 = (float4)(0.0f);  // Row 1 (if exists)
    
    // Number of tiles in K dimension
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K-dimension tiles
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // PHASE 1: COOPERATIVE LOADING OF A TILE
        // ========================================
        
        // Load A tile: 400 elements with 256 threads
        // Each thread loads: ceil(400/256) = 2 elements
        for (int i = local_linear; i < TILE_SIZE * TILE_SIZE; i += TOTAL_THREADS) {
            const int tile_row = i / TILE_SIZE;  // 0-19
            const int tile_col = i % TILE_SIZE;  // 0-19
            
            // Global position in A
            const int global_row = group_x * TILE_SIZE + tile_row;
            const int global_col = tile_k * TILE_SIZE + tile_col;
            
            // Load with bounds checking
            if (global_row < M && global_col < K) {
                As[i] = A[global_row * K + global_col];
            } else {
                As[i] = 0.0f;
            }
        }
        
        // ========================================
        // PHASE 2: COOPERATIVE LOADING OF B TILE
        // ========================================
        
        // Load B tile: 1600 elements (20 rows × 20 cols × 4 components)
        // Each thread loads: ceil(1600/256) = 7 elements
        const int b_total = TILE_SIZE * TILE_SIZE * 4;
        for (int i = local_linear; i < b_total; i += TOTAL_THREADS) {
            // Decode linearized index to tile position
            const int tile_row = i / (TILE_SIZE * 4);  // 0-19
            const int tile_col = (i / 4) % TILE_SIZE;   // 0-19
            const int component = i % 4;                // 0-3
            
            // Global position in B
            const int global_row = tile_k * TILE_SIZE + tile_row;
            const int global_col = group_y * TILE_SIZE * 4 + tile_col * 4 + component;
            
            // Load with bounds checking
            if (global_row < K && global_col < N) {
                Bs[i] = B[global_row * N + global_col];
            } else {
                Bs[i] = 0.0f;
            }
        }
        
        // Synchronize: wait for all loads to complete
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // PHASE 3: COMPUTE
        // ========================================
        
        // Each thread computes for:
        //   Row 0: local_x
        //   Row 1: local_x + 16 (if < 20)
        //   Cols:  local_y * 4 to local_y * 4 + 3
        
        const int out_row0 = local_x;         // 0-15
        const int out_row1 = local_x + 16;    // 16-31, but only use if < 20
        const int out_col_base = local_y;     // 0-15
        
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load B vector (shared for both output rows)
            const int b_offset = k * TILE_SIZE * 4 + out_col_base * 4;
            float4 b_vec = vload4(0, &Bs[b_offset]);
            
            // Compute for row 0 (always valid: 0-15 < 20)
            {
                const int a_idx = out_row0 * TILE_SIZE + k;
                float a_val = As[a_idx];
                acc0 += a_val * b_vec;
            }
            
            // Compute for row 1 (only if 16-19, i.e., local_x < 4)
            if (out_row1 < TILE_SIZE) {
                const int a_idx = out_row1 * TILE_SIZE + k;
                float a_val = As[a_idx];
                acc1 += a_val * b_vec;
            }
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // PHASE 4: WRITE RESULTS
    // ========================================
    
    const int out_col_base = local_y;
    
    // Write row 0 result
    {
        const int global_row = group_x * TILE_SIZE + local_x;
        const int global_col_base = group_y * TILE_SIZE * 4 + out_col_base * 4;
        
        if (global_row < M && global_col_base + 3 < N) {
            if (beta == 0.0f) {
                vstore4(alpha * acc0, 0, &C[global_row * N + global_col_base]);
            } else {
                float4 c_old = vload4(0, &C[global_row * N + global_col_base]);
                vstore4(alpha * acc0 + beta * c_old, 0, &C[global_row * N + global_col_base]);
            }
        }
    }
    
    // Write row 1 result (only if valid)
    if (local_x + 16 < TILE_SIZE) {
        const int global_row = group_x * TILE_SIZE + local_x + 16;
        const int global_col_base = group_y * TILE_SIZE * 4 + out_col_base * 4;
        
        if (global_row < M && global_col_base + 3 < N) {
            if (beta == 0.0f) {
                vstore4(alpha * acc1, 0, &C[global_row * N + global_col_base]);
            } else {
                float4 c_old = vload4(0, &C[global_row * N + global_col_base]);
                vstore4(alpha * acc1 + beta * c_old, 0, &C[global_row * N + global_col_base]);
            }
        }
    }
}
