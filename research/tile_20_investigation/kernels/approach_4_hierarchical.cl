/*
 * Approach 4: Hierarchical Tiling (tile=20 with sub-tiles)
 * 
 * Problem with v3:
 *   - 2048×2048: Only 335 GFLOPS (vs 651 @ 1024)
 *   - Large 20×20 tiles → high memory pressure
 *   - Working set exceeds cache capacity
 * 
 * Solution: Hierarchical tiling
 *   - Main tile: 20×20 (for output)
 *   - Sub-tiles: 10×10 (for computation)
 *   - Process 4 sub-tiles sequentially
 *   - Better cache locality
 *   - Reduced LDS pressure
 * 
 * Strategy:
 *   - 10×10 threads (100 threads, proven efficient)
 *   - Load 10×10 sub-tiles from A and B
 *   - Accumulate across K dimension
 *   - Each thread: 2×2 outputs (vectorized like v3)
 *   - Process sub-tiles in order: TL, TR, BL, BR
 * 
 * Expected: 600-700 GFLOPS @ 2048 → Average crosses 700!
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL - Approach 4 (Hierarchical)
 */

#define TILE_SIZE 20
#define SUBTILE_SIZE 10
#define LOCAL_SIZE 10
#define VEC_SIZE 4

__kernel void gemm_tile20_hierarchical(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory for 10×10 sub-tiles (smaller footprint!)
    __local float4 As[SUBTILE_SIZE * (SUBTILE_SIZE / VEC_SIZE)];  // 10×2.5 float4s
    __local float4 Bs[SUBTILE_SIZE * (SUBTILE_SIZE / VEC_SIZE)];  // 10×2.5 float4s
    
    // Thread IDs
    const int local_x = get_local_id(0);  // 0-9
    const int local_y = get_local_id(1);  // 0-9
    const int local_linear = local_y * LOCAL_SIZE + local_x;
    
    // Group IDs  
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Accumulators for 2×2 outputs (like v3)
    float acc[2][2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Number of K tiles
    const int num_k_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        
        // Process 4 sub-tiles hierarchically:
        // For a 20×20 tile, we do 2×2 sub-tiles of 10×10
        
        for (int subtile_i = 0; subtile_i < 2; subtile_i++) {      // Row sub-tiles
            for (int subtile_j = 0; subtile_j < 2; subtile_j++) {  // Col sub-tiles
                
                // ========================================
                // LOAD SUB-TILE A (10×10)
                // ========================================
                {
                    const int tile_row = local_linear / (SUBTILE_SIZE / VEC_SIZE);
                    const int tile_col_vec = local_linear % (SUBTILE_SIZE / VEC_SIZE);
                    
                    // Offset for this sub-tile within the 20×20 tile
                    const int global_row = group_x * TILE_SIZE + subtile_i * SUBTILE_SIZE + tile_row;
                    const int global_col_base = k_tile * TILE_SIZE + subtile_j * SUBTILE_SIZE + tile_col_vec * VEC_SIZE;
                    
                    if (global_row < M && global_col_base + (VEC_SIZE-1) < K) {
                        As[local_linear] = vload4(0, &A[global_row * K + global_col_base]);
                    } else if (global_row < M) {
                        float temp[VEC_SIZE];
                        for (int v = 0; v < VEC_SIZE; v++) {
                            const int gc = global_col_base + v;
                            temp[v] = (gc < K) ? A[global_row * K + gc] : 0.0f;
                        }
                        As[local_linear] = (float4)(temp[0], temp[1], temp[2], temp[3]);
                    } else {
                        As[local_linear] = (float4)(0.0f);
                    }
                }
                
                // ========================================
                // LOAD SUB-TILE B (10×10)
                // ========================================
                {
                    const int tile_row = local_linear / (SUBTILE_SIZE / VEC_SIZE);
                    const int tile_col_vec = local_linear % (SUBTILE_SIZE / VEC_SIZE);
                    
                    const int global_row = k_tile * TILE_SIZE + subtile_i * SUBTILE_SIZE + tile_row;
                    const int global_col_base = group_y * TILE_SIZE + subtile_j * SUBTILE_SIZE + tile_col_vec * VEC_SIZE;
                    
                    if (global_row < K && global_col_base + (VEC_SIZE-1) < N) {
                        Bs[local_linear] = vload4(0, &B[global_row * N + global_col_base]);
                    } else if (global_row < K) {
                        float temp[VEC_SIZE];
                        for (int v = 0; v < VEC_SIZE; v++) {
                            const int gc = global_col_base + v;
                            temp[v] = (gc < N) ? B[global_row * N + gc] : 0.0f;
                        }
                        Bs[local_linear] = (float4)(temp[0], temp[1], temp[2], temp[3]);
                    } else {
                        Bs[local_linear] = (float4)(0.0f);
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                // ========================================
                // COMPUTE on SUB-TILE (10×10)
                // ========================================
                
                #pragma unroll
                for (int k = 0; k < SUBTILE_SIZE; k++) {
                    // Load A values
                    const int a_base = k / VEC_SIZE;
                    const int a_comp = k % VEC_SIZE;
                    
                    float a_vals[2];
                    for (int row_off = 0; row_off < 2; row_off++) {
                        const int row = local_x * 2 + row_off;
                        const int a_idx = row * (SUBTILE_SIZE / VEC_SIZE) + a_base;
                        const float4 a_vec = As[a_idx];
                        
                        if (a_comp == 0) a_vals[row_off] = a_vec.s0;
                        else if (a_comp == 1) a_vals[row_off] = a_vec.s1;
                        else if (a_comp == 2) a_vals[row_off] = a_vec.s2;
                        else a_vals[row_off] = a_vec.s3;
                    }
                    
                    // Load B values
                    const int b_idx = k * (SUBTILE_SIZE / VEC_SIZE) + local_y / 2;
                    const float4 b_vec = Bs[b_idx];
                    
                    float b_vals[2];
                    if (local_y % 2 == 0) {
                        b_vals[0] = b_vec.s0;
                        b_vals[1] = b_vec.s1;
                    } else {
                        b_vals[0] = b_vec.s2;
                        b_vals[1] = b_vec.s3;
                    }
                    
                    // Accumulate
                    for (int row_off = 0; row_off < 2; row_off++) {
                        for (int col_off = 0; col_off < 2; col_off++) {
                            acc[row_off][col_off] += a_vals[row_off] * b_vals[col_off];
                        }
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
    }
    
    // ========================================
    // WRITE RESULTS
    // ========================================
    
    for (int row_off = 0; row_off < 2; row_off++) {
        for (int col_off = 0; col_off < 2; col_off++) {
            const int global_row = group_x * TILE_SIZE + local_x * 2 + row_off;
            const int global_col = group_y * TILE_SIZE + local_y * 2 + col_off;
            
            if (global_row < M && global_col < N) {
                if (beta == 0.0f) {
                    C[global_row * N + global_col] = alpha * acc[row_off][col_off];
                } else {
                    C[global_row * N + global_col] = alpha * acc[row_off][col_off] + beta * C[global_row * N + global_col];
                }
            }
        }
    }
}
