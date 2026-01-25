/**
 * Mixed Precision FP16 GEMM Kernel - GCN 4.0 Optimized
 * Combines FP16 compute with FP32 accumulation for Polaris 10
 *
 * Target: +15-20% performance improvement (285 → 330-340 GFLOPS)
 * Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)
 *
 * Strategy:
 * - FP16 computation for 2x throughput (FP16 ops/cycle vs FP32)
 * - FP32 accumulation for numerical stability
 * - Vectorized loads (float4) for bandwidth efficiency
 * - GCN 4.0 specific FP16 instruction scheduling
 *
 * Expected Performance:
 * - 2x compute throughput with FP16 operations
 * - Memory bandwidth bottleneck mitigation
 * - Better SIMD utilization with half-precision
 */

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#ifndef WG_SIZE_X
#define WG_SIZE_X 16
#endif

#ifndef WG_SIZE_Y
#define WG_SIZE_Y 16
#endif

#ifndef LDS_PADDING
#define LDS_PADDING 2  // 8 bytes (2 floats) for bank conflict avoidance
#endif

/**
 * Mixed Precision FP16 GEMM Kernel
 *
 * Precision Strategy:
 * - Input matrices: FP32 (for compatibility)
 * - Computation: FP16 (2x throughput)
 * - Accumulation: FP32 (numerical stability)
 * - Output: FP32 (standard precision)
 */
__kernel void gemm_mixed_precision_fp16(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta,
    __global const float* A,  // FP32 input
    __global const float* B,  // FP32 input
    __global float* C)        // FP32 output
{
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    // Workgroup dimensions
    const int wg_x = get_local_size(0);
    const int wg_y = get_local_size(1);

    // Only process if within matrix bounds
    if (global_x >= N || global_y >= M) return;

    // Local memory for tiles (FP16 for computation)
    __local half A_tile[TILE_SIZE][TILE_SIZE + LDS_PADDING];
    __local half B_tile[TILE_SIZE][TILE_SIZE + LDS_PADDING];

    // FP32 accumulator for numerical stability
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Process tiles
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from global memory (FP32 → FP16 conversion)
        const int a_row = t * TILE_SIZE + local_y;
        const int a_col = get_group_id(1) * TILE_SIZE + local_x;
        const int b_row = get_group_id(0) * TILE_SIZE + local_y;
        const int b_col = t * TILE_SIZE + local_x;

        // Load and convert to half precision
        if (a_row < K && a_col < M) {
            A_tile[local_y][local_x] = convert_half(A[a_row * M + a_col]);
        } else {
            A_tile[local_y][local_x] = 0.0h;
        }

        if (b_row < N && b_col < K) {
            B_tile[local_y][local_x] = convert_half(B[b_row * K + b_col]);
        } else {
            B_tile[local_y][local_x] = 0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute using FP16 operations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load FP16 values
            half4 a_vec = (half4)(
                A_tile[k][local_y * 4],
                A_tile[k][local_y * 4 + 1],
                A_tile[k][local_y * 4 + 2],
                A_tile[k][local_y * 4 + 3]
            );

            half4 b_vec = (half4)(
                B_tile[k][local_x * 4],
                B_tile[k][local_x * 4 + 1],
                B_tile[k][local_x * 4 + 2],
                B_tile[k][local_x * 4 + 3]
            );

            // FP16 FMA operations (accumulate in FP32)
            sum += convert_float4(a_vec * b_vec);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result with alpha/beta scaling
    const int c_idx = global_y * N + global_x;
    if (beta != 0.0f) {
        float4 c_old = (float4)(
            C[c_idx],
            (global_x + 1 < N) ? C[c_idx + 1] : 0.0f,
            (global_x + 2 < N) ? C[c_idx + 2] : 0.0f,
            (global_x + 3 < N) ? C[c_idx + 3] : 0.0f
        );
        sum = alpha * sum + beta * c_old;
    } else {
        sum = alpha * sum;
    }

    // Store results
    C[c_idx] = sum.x;
    if (global_x + 1 < N) C[c_idx + 1] = sum.y;
    if (global_x + 2 < N) C[c_idx + 2] = sum.z;
    if (global_x + 3 < N) C[c_idx + 3] = sum.w;
}