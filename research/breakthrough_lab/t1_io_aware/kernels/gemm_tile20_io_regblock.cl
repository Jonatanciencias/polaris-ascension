/*
 * Approach 5: Register Blocking Optimization
 * 
 * Lesson from Approach 4:
 *   - Hierarchical tiling was too complex
 *   - Indexing bugs hard to debug
 *   - Simplicity > Complexity
 * 
 * Strategy: Optimize v3 (which WORKS!)
 *   - Base: v3 structure (proven correct)
 *   - Optimization: Increase register blocking
 *   - Each thread: 4×4 = 16 outputs (vs v3's 2×2 = 4)
 *   - Better register reuse
 *   - Fewer threads (5×5 = 25 threads)
 *   - More work per thread
 * 
 * Expected:
 *   - Better performance through work efficiency
 *   - Same correctness guarantees as v3
 *   - Target: 500-600 GFLOPS @ 2048
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL - Approach 5 (Register Blocking)
 */

#define TILE_SIZE 20
#define LOCAL_SIZE 5   // 5×5 = 25 threads (vs v3's 10×10 = 100)
#define VEC_SIZE 4

__kernel void gemm_tile20_io_regblock(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory for 20×20 tiles (same as v3)
    __local float4 As[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];
    __local float4 Bs[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];
    
    // Thread IDs
    const int local_x = get_local_id(0);  // 0-4
    const int local_y = get_local_id(1);  // 0-4
    const int local_linear = local_y * LOCAL_SIZE + local_x;  // 0-24
    
    // Group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Each thread computes 4×4 = 16 outputs (register blocking!)
    float acc[4][4];  // [row_offset][col_offset]
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Number of tiles in K
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K tiles
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // COOPERATIVE LOADING with FLOAT4
        // 100 float4s with 25 threads → each loads 4
        // ========================================
        
        const int lds_size = TILE_SIZE * (TILE_SIZE / VEC_SIZE);  // 100
        
        for (int i = local_linear; i < lds_size; i += LOCAL_SIZE * LOCAL_SIZE) {
            // Load A
            const int tile_row = i / (TILE_SIZE / VEC_SIZE);
            const int tile_col_vec = i % (TILE_SIZE / VEC_SIZE);
            const int global_row = group_x * TILE_SIZE + tile_row;
            const int global_col_base = tile_k * TILE_SIZE + tile_col_vec * VEC_SIZE;
            
            if (global_row < M && global_col_base + (VEC_SIZE-1) < K) {
                As[i] = vload4(0, &A[global_row * K + global_col_base]);
            } else if (global_row < M) {
                float temp[VEC_SIZE];
                for (int v = 0; v < VEC_SIZE; v++) {
                    const int gc = global_col_base + v;
                    temp[v] = (gc < K) ? A[global_row * K + gc] : 0.0f;
                }
                As[i] = (float4)(temp[0], temp[1], temp[2], temp[3]);
            } else {
                As[i] = (float4)(0.0f);
            }
            
            // Load B
            const int global_row_b = tile_k * TILE_SIZE + tile_row;
            const int global_col_base_b = group_y * TILE_SIZE + tile_col_vec * VEC_SIZE;
            
            if (global_row_b < K && global_col_base_b + (VEC_SIZE-1) < N) {
                Bs[i] = vload4(0, &B[global_row_b * N + global_col_base_b]);
            } else if (global_row_b < K) {
                float temp[VEC_SIZE];
                for (int v = 0; v < VEC_SIZE; v++) {
                    const int gc = global_col_base_b + v;
                    temp[v] = (gc < N) ? B[global_row_b * N + gc] : 0.0f;
                }
                Bs[i] = (float4)(temp[0], temp[1], temp[2], temp[3]);
            } else {
                Bs[i] = (float4)(0.0f);
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // COMPUTE: 4 rows × 4 cols (register blocking)
        // Thread (local_x, local_y) computes:
        //   Rows: local_x*4 to local_x*4+3
        //   Cols: local_y*4 to local_y*4+3
        // ========================================
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A values for 4 rows
            const int a_base = k / VEC_SIZE;
            const int a_comp = k % VEC_SIZE;
            
            float a_vals[4];
            for (int row_off = 0; row_off < 4; row_off++) {
                const int row = local_x * 4 + row_off;
                const int a_idx = row * (TILE_SIZE / VEC_SIZE) + a_base;
                const float4 a_vec = As[a_idx];
                
                if (a_comp == 0) a_vals[row_off] = a_vec.s0;
                else if (a_comp == 1) a_vals[row_off] = a_vec.s1;
                else if (a_comp == 2) a_vals[row_off] = a_vec.s2;
                else a_vals[row_off] = a_vec.s3;
            }
            
            // Load B values for 4 cols
            const int b_idx = k * (TILE_SIZE / VEC_SIZE) + local_y;
            const float4 b_vec = Bs[b_idx];
            
            // Compute 4×4 = 16 elements
            #pragma unroll
            for (int row_off = 0; row_off < 4; row_off++) {
                #pragma unroll
                for (int col_off = 0; col_off < 4; col_off++) {
                    float b_val;
                    if (col_off == 0) b_val = b_vec.s0;
                    else if (col_off == 1) b_val = b_vec.s1;
                    else if (col_off == 2) b_val = b_vec.s2;
                    else b_val = b_vec.s3;
                    
                    acc[row_off][col_off] += a_vals[row_off] * b_val;
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // WRITE RESULTS: 4×4 outputs per thread
    // ========================================
    
    for (int row_off = 0; row_off < 4; row_off++) {
        for (int col_off = 0; col_off < 4; col_off++) {
            const int global_row = group_x * TILE_SIZE + local_x * 4 + row_off;
            const int global_col = group_y * TILE_SIZE + local_y * 4 + col_off;
            
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
