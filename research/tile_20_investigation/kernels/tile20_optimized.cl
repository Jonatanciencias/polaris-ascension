/*
 * tile20 Optimized for Large Matrices (2048+)
 * 
 * Problem Analysis:
 *   - tile20 @ 2048: 335 GFLOPS (vs 601 @ 1024)
 *   - Root cause: Cache thrashing, not memory latency
 *   - Prefetching failed: no real async in OpenCL 1.1/Clover
 * 
 * New Strategy: Improved Memory Access Pattern
 *   1. Better coalescing: Sequential access to global memory
 *   2. Reduced LDS bank conflicts: Padding
 *   3. Optimized unrolling: Better instruction scheduling
 *   4. Smaller register pressure: Incremental accumulation
 * 
 * Expected: 335 → 400-450 GFLOPS @ 2048
 * 
 * Phase 1 - Step 1 (v2): Memory-Optimized Approach
 * Date: 4 febrero 2026
 */

#define TILE_SIZE 20
#define LOCAL_SIZE 10
#define VEC_SIZE 4
#define LDS_PAD 1  // Padding to avoid bank conflicts

__kernel void gemm_tile20_optimized(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory with padding to reduce bank conflicts
    __local float As[TILE_SIZE][TILE_SIZE + LDS_PAD];
    __local float Bs[TILE_SIZE][TILE_SIZE + LDS_PAD];
    
    // Work-item indices
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Global indices for output
    const int out_row_base = group_x * TILE_SIZE + local_x * 2;
    const int out_col_base = group_y * TILE_SIZE + local_y * 2;
    
    // Accumulators for 2×2 output block
    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // ========================================
    // MAIN LOOP: Process tiles along K
    // ========================================
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // LOAD TILE A (coalesced, 2 loads per thread)
        // ========================================
        // Each thread loads 2 elements from A
        {
            const int global_row = group_x * TILE_SIZE + local_x * 2;
            const int global_col_base = tile_k * TILE_SIZE;
            
            // Load 2 rows worth of data (20 elements total across 10 threads)
            #pragma unroll 2
            for (int i = 0; i < 2; i++) {
                const int gr = global_row + i;
                
                // Each thread loads 2 consecutive elements
                #pragma unroll 2
                for (int j = 0; j < 2; j++) {
                    const int local_col = local_y * 2 + j;
                    const int gc = global_col_base + local_col;
                    
                    if (gr < M && gc < K) {
                        As[local_x * 2 + i][local_col] = A[gr * K + gc];
                    } else {
                        As[local_x * 2 + i][local_col] = 0.0f;
                    }
                }
            }
        }
        
        // ========================================
        // LOAD TILE B (coalesced, 2 loads per thread)
        // ========================================
        {
            const int global_row_base = tile_k * TILE_SIZE;
            const int global_col = group_y * TILE_SIZE + local_y * 2;
            
            // Load 2 columns worth of data
            #pragma unroll 2
            for (int j = 0; j < 2; j++) {
                const int gc = global_col + j;
                
                // Each thread loads 2 consecutive elements
                #pragma unroll 2
                for (int i = 0; i < 2; i++) {
                    const int local_row = local_x * 2 + i;
                    const int gr = global_row_base + local_row;
                    
                    if (gr < K && gc < N) {
                        Bs[local_row][local_y * 2 + j] = B[gr * N + gc];
                    } else {
                        Bs[local_row][local_y * 2 + j] = 0.0f;
                    }
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // COMPUTE: 2×2 output block
        // ========================================
        // Unroll aggressively for better instruction scheduling
        
        #pragma unroll 10
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A values (2 rows)
            float a0 = As[local_x * 2][k];
            float a1 = As[local_x * 2 + 1][k];
            
            // Load B values (2 cols)
            float b0 = Bs[k][local_y * 2];
            float b1 = Bs[k][local_y * 2 + 1];
            
            // Compute 2×2 block (4 FMAs)
            acc00 = mad(a0, b0, acc00);
            acc01 = mad(a0, b1, acc01);
            acc10 = mad(a1, b0, acc10);
            acc11 = mad(a1, b1, acc11);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // WRITE RESULTS (coalesced)
    // ========================================
    
    // Write 2×2 block
    if (out_row_base < M && out_col_base < N) {
        const int idx00 = out_row_base * N + out_col_base;
        C[idx00] = (beta == 0.0f) ? (alpha * acc00) : mad(alpha, acc00, beta * C[idx00]);
    }
    
    if (out_row_base < M && out_col_base + 1 < N) {
        const int idx01 = out_row_base * N + out_col_base + 1;
        C[idx01] = (beta == 0.0f) ? (alpha * acc01) : mad(alpha, acc01, beta * C[idx01]);
    }
    
    if (out_row_base + 1 < M && out_col_base < N) {
        const int idx10 = (out_row_base + 1) * N + out_col_base;
        C[idx10] = (beta == 0.0f) ? (alpha * acc10) : mad(alpha, acc10, beta * C[idx10]);
    }
    
    if (out_row_base + 1 < M && out_col_base + 1 < N) {
        const int idx11 = (out_row_base + 1) * N + out_col_base + 1;
        C[idx11] = (beta == 0.0f) ? (alpha * acc11) : mad(alpha, acc11, beta * C[idx11]);
    }
}
