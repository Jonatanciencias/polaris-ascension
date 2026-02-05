/*
 * Tile=24 Vectorized Implementation - Phase 2.1 Step 2
 * 
 * Design Rationale:
 *   - tile20 v3: 10×10 threads (100) → 819.7 GFLOPS @ 1400
 *   - tile16 baseline: 16×16 threads (256) → 143.6 GFLOPS
 *   - tile24: 12×12 threads (144) → target 850-900 GFLOPS
 * 
 * Key Innovation: Sweet Spot Threading
 *   - 144 threads: More compute than tile20 (100), less overhead than tile16 (256)
 *   - 24×24 tile: 20% more work per tile than tile20
 *   - float4 vectorization: Maintain memory bandwidth efficiency
 *   - Each thread: 2×2 sub-tile (4 elements)
 * 
 * Thread Layout:
 *   Workgroup: 12×12 = 144 threads
 *   Coverage: Each thread handles 2 rows × 2 columns
 *   Total: 12×2 = 24 rows, 12×2 = 24 columns ✓
 * 
 * Memory Strategy:
 *   - LDS size: 24×24×4 bytes = 2304 bytes per tile (2 tiles = 4.6 KB)
 *   - Well below 32 KB LDS limit
 *   - float4 vectorization for coalesced access
 *   - Bank conflict avoidance through proper stride
 * 
 * Expected Performance:
 *   - @1024: 650-700 GFLOPS
 *   - @1400: 850-900 GFLOPS (target)
 *   - @2048: 400-450 GFLOPS
 * 
 * Author: Phase 2.1 - Quick Wins
 * Date: February 4, 2026
 * Status: PRODUCTION CANDIDATE
 */

#define TILE_SIZE 24
#define LOCAL_SIZE 12
#define TOTAL_THREADS (LOCAL_SIZE * LOCAL_SIZE)  // 144
#define VEC_SIZE 4
#define ELEMENTS_PER_THREAD 4  // 2×2 sub-tile

__kernel void gemm_tile24_vectorized(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory for 24×24 tiles
    // Use float4 to optimize memory access (24×6 float4s = 24×24 floats)
    __local float4 As[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];  
    __local float4 Bs[TILE_SIZE * (TILE_SIZE / VEC_SIZE)];  
    
    // Thread IDs
    const int local_x = get_local_id(0);  // 0-11
    const int local_y = get_local_id(1);  // 0-11
    const int local_linear = local_y * LOCAL_SIZE + local_x;  // 0-143
    
    // Group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Each thread computes 2 rows × 2 columns
    // Thread (tx, ty) → rows [ty*2, ty*2+1], cols [tx*2, tx*2+1]
    const int c_row_start = group_y * TILE_SIZE + local_y * 2;
    const int c_col_start = group_x * TILE_SIZE + local_x * 2;
    
    // Accumulator: 2×2 sub-matrix
    float acc[2][2];
    acc[0][0] = 0.0f;
    acc[0][1] = 0.0f;
    acc[1][0] = 0.0f;
    acc[1][1] = 0.0f;
    
    // Number of tiles to process
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Tile start in K dimension
        const int k_start = tile_idx * TILE_SIZE;
        
        // ================================================================
        // COOPERATIVE LOADING - A tile (M×K section)
        // ================================================================
        // 144 threads load 24×24 = 576 elements
        // Each thread loads 576/144 = 4 elements (1 float4)
        
        // Strategy: Linear assignment
        // Thread i loads element i, i+144, i+288, i+432
        // This gives perfect coverage: 4×144 = 576 ✓
        
        #pragma unroll
        for (int load_idx = 0; load_idx < 4; load_idx++) {
            const int elem_idx = local_linear + load_idx * TOTAL_THREADS;
            
            if (elem_idx < TILE_SIZE * (TILE_SIZE / VEC_SIZE)) {
                const int tile_row = elem_idx / (TILE_SIZE / VEC_SIZE);
                const int tile_col_vec = elem_idx % (TILE_SIZE / VEC_SIZE);
                
                const int global_row = group_y * TILE_SIZE + tile_row;
                const int global_col = k_start + tile_col_vec * VEC_SIZE;
                
                // Bounds check
                if (global_row < M && global_col < K) {
                    const int a_offset = global_row * K + global_col;
                    
                    // Load float4 if all 4 elements are in bounds
                    if (global_col + 3 < K) {
                        As[elem_idx] = vload4(0, A + a_offset);
                    } else {
                        // Scalar fallback for boundary
                        float4 temp = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                        for (int i = 0; i < VEC_SIZE && global_col + i < K; i++) {
                            temp[i] = A[a_offset + i];
                        }
                        As[elem_idx] = temp;
                    }
                } else {
                    As[elem_idx] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        
        // ================================================================
        // COOPERATIVE LOADING - B tile (K×N section)
        // ================================================================
        #pragma unroll
        for (int load_idx = 0; load_idx < 4; load_idx++) {
            const int elem_idx = local_linear + load_idx * TOTAL_THREADS;
            
            if (elem_idx < TILE_SIZE * (TILE_SIZE / VEC_SIZE)) {
                const int tile_row = elem_idx / (TILE_SIZE / VEC_SIZE);
                const int tile_col_vec = elem_idx % (TILE_SIZE / VEC_SIZE);
                
                const int global_row = k_start + tile_row;
                const int global_col = group_x * TILE_SIZE + tile_col_vec * VEC_SIZE;
                
                // Bounds check
                if (global_row < K && global_col < N) {
                    const int b_offset = global_row * N + global_col;
                    
                    // Load float4 if all 4 elements are in bounds
                    if (global_col + 3 < N) {
                        Bs[elem_idx] = vload4(0, B + b_offset);
                    } else {
                        // Scalar fallback for boundary
                        float4 temp = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                        for (int i = 0; i < VEC_SIZE && global_col + i < N; i++) {
                            temp[i] = B[b_offset + i];
                        }
                        Bs[elem_idx] = temp;
                    }
                } else {
                    Bs[elem_idx] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        
        // Synchronize: all threads must complete loading
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ================================================================
        // COMPUTATION - 2×2 sub-tile
        // ================================================================
        // Thread (local_x, local_y) computes rows [local_y*2, local_y*2+1]
        //                                   cols [local_x*2, local_x*2+1]
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A elements for this thread's rows
            const int a_row0_vec_idx = (local_y * 2) * (TILE_SIZE / VEC_SIZE) + k / VEC_SIZE;
            const int a_row1_vec_idx = (local_y * 2 + 1) * (TILE_SIZE / VEC_SIZE) + k / VEC_SIZE;
            const int a_elem_offset = k % VEC_SIZE;
            
            float a00 = As[a_row0_vec_idx][a_elem_offset];
            float a10 = As[a_row1_vec_idx][a_elem_offset];
            
            // Load B elements for this thread's columns
            const int b_col_vec_idx = k * (TILE_SIZE / VEC_SIZE) + (local_x * 2) / VEC_SIZE;
            const int b_col0_offset = (local_x * 2) % VEC_SIZE;
            const int b_col1_offset = (local_x * 2 + 1) % VEC_SIZE;
            
            float b00 = Bs[b_col_vec_idx][b_col0_offset];
            
            // Handle b01 - may be in same or next float4
            float b01;
            if (b_col1_offset < VEC_SIZE) {
                b01 = Bs[b_col_vec_idx][b_col1_offset];
            } else {
                b01 = Bs[b_col_vec_idx + 1][0];
            }
            
            // Compute 2×2 outer product
            acc[0][0] += a00 * b00;
            acc[0][1] += a00 * b01;
            acc[1][0] += a10 * b00;
            acc[1][1] += a10 * b01;
        }
        
        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ================================================================
    // WRITE RESULTS - 2×2 sub-tile to global memory
    // ================================================================
    const int c_row0 = c_row_start;
    const int c_row1 = c_row_start + 1;
    const int c_col0 = c_col_start;
    const int c_col1 = c_col_start + 1;
    
    // Write with bounds check
    if (c_row0 < M && c_col0 < N) {
        const int idx = c_row0 * N + c_col0;
        C[idx] = alpha * acc[0][0] + beta * C[idx];
    }
    
    if (c_row0 < M && c_col1 < N) {
        const int idx = c_row0 * N + c_col1;
        C[idx] = alpha * acc[0][1] + beta * C[idx];
    }
    
    if (c_row1 < M && c_col0 < N) {
        const int idx = c_row1 * N + c_col0;
        C[idx] = alpha * acc[1][0] + beta * C[idx];
    }
    
    if (c_row1 < M && c_col1 < N) {
        const int idx = c_row1 * N + c_col1;
        C[idx] = alpha * acc[1][1] + beta * C[idx];
    }
}
