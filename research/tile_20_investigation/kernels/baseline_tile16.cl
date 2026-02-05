/*
 * Baseline GEMM kernel - tile=16 (FLOAT4_VEC)
 * Copied from production for Phase 1 testing
 */

#define TILE_SIZE 16

__kernel void gemm_float4_vec(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    // Local memory tiles
    __local float As[TILE_SIZE * TILE_SIZE];
    __local float Bs[TILE_SIZE * TILE_SIZE * 4];  // 4× for vectorization
    
    // Work-item indices
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_row = get_global_id(0);
    const int global_col_base = get_global_id(1) * 4;  // 4 columns per work-item
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);
    
    // Accumulator - 4 values for 4 columns
    float4 sum = (float4)(0.0f);
    
    // Number of tiles
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K in tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A
        const int a_row = group_row * TILE_SIZE + local_row;
        const int a_col = t * TILE_SIZE + local_col;
        
        if (a_row < M && a_col < K) {
            As[local_row * TILE_SIZE + local_col] = A[a_row * K + a_col];
        } else {
            As[local_row * TILE_SIZE + local_col] = 0.0f;
        }
        
        // Load tile from B (4 consecutive elements per work-item)
        const int b_row = t * TILE_SIZE + local_row;
        const int b_col_base = group_col * TILE_SIZE * 4 + local_col * 4;
        const int lds_offset = local_row * TILE_SIZE * 4 + local_col * 4;
        
        if (b_row < K && b_col_base + 3 < N) {
            // Vectorized load
            float4 b_vec = vload4(0, &B[b_row * N + b_col_base]);
            Bs[lds_offset + 0] = b_vec.x;
            Bs[lds_offset + 1] = b_vec.y;
            Bs[lds_offset + 2] = b_vec.z;
            Bs[lds_offset + 3] = b_vec.w;
        } else if (b_row < K) {
            // Boundary
            for (int i = 0; i < 4; i++) {
                if (b_col_base + i < N) {
                    Bs[lds_offset + i] = B[b_row * N + b_col_base + i];
                } else {
                    Bs[lds_offset + i] = 0.0f;
                }
            }
        } else {
            Bs[lds_offset + 0] = 0.0f;
            Bs[lds_offset + 1] = 0.0f;
            Bs[lds_offset + 2] = 0.0f;
            Bs[lds_offset + 3] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_val = As[local_row * TILE_SIZE + k];
            
            const int b_offset = k * TILE_SIZE * 4 + local_col * 4;
            float4 b_vec;
            b_vec.x = Bs[b_offset + 0];
            b_vec.y = Bs[b_offset + 1];
            b_vec.z = Bs[b_offset + 2];
            b_vec.w = Bs[b_offset + 3];
            
            // Vectorized FMA
            sum = mad(a_val, b_vec, sum);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    if (global_row < M && global_col_base + 3 < N) {
        const int c_idx = global_row * N + global_col_base;
        float4 c_vec = sum;
        vstore4(c_vec, 0, &C[c_idx]);
    } else if (global_row < M) {
        // Boundary
        for (int i = 0; i < 4 && global_col_base + i < N; i++) {
            const int c_idx = global_row * N + global_col_base + i;
            float val = (i == 0) ? sum.x : (i == 1) ? sum.y : (i == 2) ? sum.z : sum.w;
            C[c_idx] = val;
        }
    }
}
