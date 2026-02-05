/*
 * Approach 2 v3: Vectorized (10×10 threads with FLOAT4)
 * 
 * Key Insight from v1 & v2:
 *   - v1 (10×10): 554 GFLOPS ✅ (thread efficiency: 5.54 GFLOPS/thread)
 *   - v2 (20×10): 501 GFLOPS ✅ (thread efficiency: 2.50 GFLOPS/thread)
 *   - MORE threads ≠ BETTER performance!
 * 
 * Solution: Vectorization instead of more threads
 *   - Keep 10×10 = 100 threads (proven efficient)
 *   - Use float4 for better memory bandwidth (like FLOAT4_VEC)
 *   - Each thread: 2 rows × 2 columns BUT using float4 internally
 *   - Better memory coalescing
 * 
 * Expected: 600-750 GFLOPS (like production FLOAT4_VEC)
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL - Approach 2 v3 (Vectorized)
 */

#define TILE_SIZE 20
#define LOCAL_SIZE 10
#define TOTAL_THREADS (LOCAL_SIZE * LOCAL_SIZE)  // 100
#define VEC_SIZE 4

__kernel void gemm_tile20_vectorized(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory for 20×20 tiles
    __local float4 As[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];  // 20×5 float4s = 20×20 floats
    __local float4 Bs[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];  // 20×5 float4s = 20×20 floats
    
    // Thread IDs
    const int local_x = get_local_id(0);  // 0-9
    const int local_y = get_local_id(1);  // 0-9
    const int local_linear = local_y * LOCAL_SIZE + local_x;  // 0-99
    
    // Group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Each thread computes 2 rows × 2 columns
    float acc[2][2];  // [row_offset][col_offset]
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Number of tiles in K
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K tiles
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // COOPERATIVE LOADING with FLOAT4
        // ========================================
        
        // Load A: 100 float4s (20×20 floats) with 100 threads → each loads 1 float4
        {
            const int tile_row = local_linear / (TILE_SIZE / VEC_SIZE);
            const int tile_col_vec = local_linear % (TILE_SIZE / VEC_SIZE);
            const int global_row = group_x * TILE_SIZE + tile_row;
            const int global_col_base = tile_k * TILE_SIZE + tile_col_vec * VEC_SIZE;
            
            if (global_row < M && global_col_base + (VEC_SIZE-1) < K) {
                As[local_linear] = vload4(0, &A[global_row * K + global_col_base]);
            } else if (global_row < M) {
                // Partial load
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
        
        // Load B: 100 float4s (20×20 floats) with 100 threads → each loads 1 float4
        {
            const int tile_row = local_linear / (TILE_SIZE / VEC_SIZE);
            const int tile_col_vec = local_linear % (TILE_SIZE / VEC_SIZE);
            const int global_row = tile_k * TILE_SIZE + tile_row;
            const int global_col_base = group_y * TILE_SIZE + tile_col_vec * VEC_SIZE;
            
            if (global_row < K && global_col_base + (VEC_SIZE-1) < N) {
                Bs[local_linear] = vload4(0, &B[global_row * N + global_col_base]);
            } else if (global_row < K) {
                // Partial load
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
        // COMPUTE: 2 rows × 2 cols using float4
        // ========================================
        
        // Thread (local_x, local_y) computes:
        //   Rows: local_x*2, local_x*2+1
        //   Cols: local_y*2, local_y*2+1
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A values for both rows (need individual components)
            const int a_base = k / VEC_SIZE;
            const int a_comp = k % VEC_SIZE;
            
            float a_vals[2];
            for (int row_off = 0; row_off < 2; row_off++) {
                const int row = local_x * 2 + row_off;
                const int a_idx = row * (TILE_SIZE / VEC_SIZE) + a_base;
                const float4 a_vec = As[a_idx];
                
                if (a_comp == 0) a_vals[row_off] = a_vec.s0;
                else if (a_comp == 1) a_vals[row_off] = a_vec.s1;
                else if (a_comp == 2) a_vals[row_off] = a_vec.s2;
                else a_vals[row_off] = a_vec.s3;
            }
            
            // Load B values (need individual components for 2 cols)
            const int b_idx = k * (TILE_SIZE / VEC_SIZE) + local_y / 2;
            const float4 b_vec = Bs[b_idx];
            
            float b_vals[2];
            if (local_y % 2 == 0) {
                // Even local_y → use .s0 and .s1
                b_vals[0] = b_vec.s0;
                b_vals[1] = b_vec.s1;
            } else {
                // Odd local_y → use .s2 and .s3
                b_vals[0] = b_vec.s2;
                b_vals[1] = b_vec.s3;
            }
            
            // Compute
            for (int row_off = 0; row_off < 2; row_off++) {
                for (int col_off = 0; col_off < 2; col_off++) {
                    acc[row_off][col_off] += a_vals[row_off] * b_vals[col_off];
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
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
