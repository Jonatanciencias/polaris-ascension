/*
 * tile20 with float8 Vectorization - Phase 2.1 Extension
 * 
 * STRATEGY: Leverage float8 for maximum memory bandwidth
 * 
 * Current best (float4):
 *   - tile20 @ 1400: 866.9 GFLOPS (peak)
 *   - Vectorization: float4 (128-bit loads)
 *   - Memory bandwidth: ~50-60% utilized
 * 
 * float8 Potential:
 *   - Vectorization: float8 (256-bit loads)
 *   - Double memory bandwidth per instruction
 *   - Better coalescing (fewer transactions)
 *   - Trade-off: Higher register pressure
 * 
 * Target: 900-1000 GFLOPS @ 1400
 * Expected: +4-15% vs float4 (866.9 → 900-1000)
 * 
 * Design:
 *   - Workgroup: 10×10 = 100 threads (same as tile20)
 *   - Tile size: 20×20
 *   - Each thread: 2×2 output elements
 *   - Vectorization: float8 for A/B loads
 *   - LDS: Padded to avoid bank conflicts
 * 
 * Innovation: Load 8 elements at once instead of 4
 *   - A/B tiles: Load in float8 chunks
 *   - Compute loop: Process 8 K-elements per iteration
 *   - Unroll factor: 8 (to match vector width)
 * 
 * Date: 4 febrero 2026
 * Status: EXPERIMENTAL (testing float8 viability)
 */

#define TILE_SIZE 20
#define LOCAL_SIZE 10  // 10×10 = 100 threads
#define VEC8_SIZE 8
#define LDS_PAD 1  // Avoid bank conflicts

__kernel void gemm_tile20_float8(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // LDS with padding
    __local float As[TILE_SIZE][TILE_SIZE + LDS_PAD];
    __local float Bs[TILE_SIZE][TILE_SIZE + LDS_PAD];
    
    // Thread indices
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Output indices (each thread → 2×2 block)
    const int out_row_base = group_x * TILE_SIZE + local_x * 2;
    const int out_col_base = group_y * TILE_SIZE + local_y * 2;
    
    // Accumulators (2×2 output block)
    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // ========================================
    // MAIN LOOP: Process K dimension
    // ========================================
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // LOAD TILE A using float8 vectorization
        // ========================================
        // Strategy: Each thread loads 2 rows, using float8 for 8 columns at once
        {
            const int global_row = group_x * TILE_SIZE + local_x * 2;
            const int global_col_base = tile_k * TILE_SIZE;
            
            // Load 2 rows (each thread handles 2 rows × 20 cols)
            #pragma unroll 2
            for (int i = 0; i < 2; i++) {
                const int gr = global_row + i;
                const int local_row = local_x * 2 + i;
                
                if (gr < M && global_col_base < K) {
                    // Try float8 load for first 8 elements
                    if (global_col_base + 7 < K && local_y == 0) {
                        float8 vec_a = vload8(0, &A[gr * K + global_col_base]);
                        As[local_row][0] = vec_a.s0;
                        As[local_row][1] = vec_a.s1;
                        As[local_row][2] = vec_a.s2;
                        As[local_row][3] = vec_a.s3;
                        As[local_row][4] = vec_a.s4;
                        As[local_row][5] = vec_a.s5;
                        As[local_row][6] = vec_a.s6;
                        As[local_row][7] = vec_a.s7;
                    }
                    
                    // Try float8 load for next 8 elements (cols 8-15)
                    if (global_col_base + 15 < K && local_y == 1) {
                        float8 vec_a = vload8(0, &A[gr * K + global_col_base + 8]);
                        As[local_row][8] = vec_a.s0;
                        As[local_row][9] = vec_a.s1;
                        As[local_row][10] = vec_a.s2;
                        As[local_row][11] = vec_a.s3;
                        As[local_row][12] = vec_a.s4;
                        As[local_row][13] = vec_a.s5;
                        As[local_row][14] = vec_a.s6;
                        As[local_row][15] = vec_a.s7;
                    }
                    
                    // Scalar loads for remaining 4 elements (cols 16-19)
                    if (local_y >= 2 && local_y <= 5) {
                        int col_idx = 16 + (local_y - 2);
                        int gc = global_col_base + col_idx;
                        if (gc < K) {
                            As[local_row][col_idx] = A[gr * K + gc];
                        } else {
                            As[local_row][col_idx] = 0.0f;
                        }
                    }
                } else {
                    // Out of bounds - zero out
                    #pragma unroll 2
                    for (int j = 0; j < 2; j++) {
                        int col_idx = local_y * 2 + j;
                        if (col_idx < TILE_SIZE) {
                            As[local_row][col_idx] = 0.0f;
                        }
                    }
                }
            }
        }
        
        // ========================================
        // LOAD TILE B using float8 vectorization
        // ========================================
        {
            const int global_row_base = tile_k * TILE_SIZE;
            const int global_col = group_y * TILE_SIZE + local_y * 2;
            
            // Load 2 columns (each thread handles 20 rows × 2 cols)
            #pragma unroll 2
            for (int j = 0; j < 2; j++) {
                const int gc = global_col + j;
                const int local_col = local_y * 2 + j;
                
                if (gc < N && global_row_base < K) {
                    // Try float8 load for first 8 rows
                    if (global_row_base + 7 < K && local_x == 0) {
                        // Load 8 consecutive elements from column gc
                        float8 vec_b;
                        #pragma unroll 8
                        for (int k = 0; k < 8; k++) {
                            int gr = global_row_base + k;
                            if (gr < K) {
                                ((float*)&vec_b)[k] = B[gr * N + gc];
                            } else {
                                ((float*)&vec_b)[k] = 0.0f;
                            }
                        }
                        Bs[0][local_col] = vec_b.s0;
                        Bs[1][local_col] = vec_b.s1;
                        Bs[2][local_col] = vec_b.s2;
                        Bs[3][local_col] = vec_b.s3;
                        Bs[4][local_col] = vec_b.s4;
                        Bs[5][local_col] = vec_b.s5;
                        Bs[6][local_col] = vec_b.s6;
                        Bs[7][local_col] = vec_b.s7;
                    }
                    
                    // Try float8 load for next 8 rows (rows 8-15)
                    if (global_row_base + 15 < K && local_x == 1) {
                        float8 vec_b;
                        #pragma unroll 8
                        for (int k = 0; k < 8; k++) {
                            int gr = global_row_base + 8 + k;
                            if (gr < K) {
                                ((float*)&vec_b)[k] = B[gr * N + gc];
                            } else {
                                ((float*)&vec_b)[k] = 0.0f;
                            }
                        }
                        Bs[8][local_col] = vec_b.s0;
                        Bs[9][local_col] = vec_b.s1;
                        Bs[10][local_col] = vec_b.s2;
                        Bs[11][local_col] = vec_b.s3;
                        Bs[12][local_col] = vec_b.s4;
                        Bs[13][local_col] = vec_b.s5;
                        Bs[14][local_col] = vec_b.s6;
                        Bs[15][local_col] = vec_b.s7;
                    }
                    
                    // Scalar loads for remaining 4 rows (rows 16-19)
                    if (local_x >= 2 && local_x <= 5) {
                        int row_idx = 16 + (local_x - 2);
                        int gr = global_row_base + row_idx;
                        if (gr < K) {
                            Bs[row_idx][local_col] = B[gr * N + gc];
                        } else {
                            Bs[row_idx][local_col] = 0.0f;
                        }
                    }
                } else {
                    // Out of bounds
                    #pragma unroll 2
                    for (int i = 0; i < 2; i++) {
                        int row_idx = local_x * 2 + i;
                        if (row_idx < TILE_SIZE) {
                            Bs[row_idx][local_col] = 0.0f;
                        }
                    }
                }
            }
        }
        
        // Synchronize to ensure tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // COMPUTE: MAD operations on LDS tiles
        // ========================================
        // Process 8 K-elements at a time (match float8 width)
        #pragma unroll 2
        for (int kk = 0; kk < TILE_SIZE; kk += 8) {
            // Unroll 8 iterations to match vector width
            #pragma unroll 8
            for (int k = 0; k < 8; k++) {
                if (kk + k < TILE_SIZE) {
                    // Load A values (2 rows)
                    const float a0 = As[local_x * 2 + 0][kk + k];
                    const float a1 = As[local_x * 2 + 1][kk + k];
                    
                    // Load B values (2 cols)
                    const float b0 = Bs[kk + k][local_y * 2 + 0];
                    const float b1 = Bs[kk + k][local_y * 2 + 1];
                    
                    // Compute 2×2 block
                    acc00 += a0 * b0;
                    acc01 += a0 * b1;
                    acc10 += a1 * b0;
                    acc11 += a1 * b1;
                }
            }
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // WRITE OUTPUT (with alpha/beta scaling)
    // ========================================
    {
        const int out_row0 = out_row_base;
        const int out_row1 = out_row_base + 1;
        const int out_col0 = out_col_base;
        const int out_col1 = out_col_base + 1;
        
        // Write 2×2 output block
        if (out_row0 < M && out_col0 < N) {
            const int idx = out_row0 * N + out_col0;
            C[idx] = alpha * acc00 + beta * C[idx];
        }
        if (out_row0 < M && out_col1 < N) {
            const int idx = out_row0 * N + out_col1;
            C[idx] = alpha * acc01 + beta * C[idx];
        }
        if (out_row1 < M && out_col0 < N) {
            const int idx = out_row1 * N + out_col0;
            C[idx] = alpha * acc10 + beta * C[idx];
        }
        if (out_row1 < M && out_col1 < N) {
            const int idx = out_row1 * N + out_col1;
            C[idx] = alpha * acc11 + beta * C[idx];
        }
    }
}

/*
 * EXPECTED PERFORMANCE:
 * 
 * float4 baseline:
 *   @ 1400: 866.9 GFLOPS
 *   @ 2048: 331.6 GFLOPS
 * 
 * float8 target (conservative +10%):
 *   @ 1400: 950 GFLOPS (+83 GFLOPS)
 *   @ 2048: 365 GFLOPS (+33 GFLOPS)
 * 
 * float8 optimistic (+15%):
 *   @ 1400: 997 GFLOPS (+130 GFLOPS)
 *   @ 2048: 381 GFLOPS (+49 GFLOPS)
 * 
 * Trade-offs:
 *   ✅ 2× memory bandwidth per load instruction
 *   ✅ Better coalescing (fewer memory transactions)
 *   ✅ Potential for better instruction scheduling
 *   ⚠️  Higher register pressure (may spill)
 *   ⚠️  More complex loading logic
 *   ⚠️  May hit diminishing returns (bandwidth not bottleneck)
 * 
 * If float8 FAILS (< 900 GFLOPS):
 *   - Reason: Register spilling, or bandwidth not the bottleneck
 *   - Action: Discard, use float4 (866.9 GFLOPS is excellent)
 *   - Time lost: 2-3 hours (acceptable risk)
 * 
 * If float8 SUCCEEDS (>= 950 GFLOPS):
 *   - Benefit: +83-130 GFLOPS peak improvement
 *   - Action: Integrate as primary kernel
 *   - New peak: 950-1000 GFLOPS
 */
