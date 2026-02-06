/**
 * Debug kernel - just copy A to C to test loading
 */

#ifndef TILE_SIZE
#define TILE_SIZE 8
#endif

__kernel void gemm_debug_copy(
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // Just copy A to C for debugging
    int global_x = wg_x * TILE_SIZE + local_x;
    int global_y = wg_y * TILE_SIZE + local_y;

    if (global_x < M && global_y < N) {
        C[global_y * N + global_x] = A[global_y * K + global_x];
    }
}