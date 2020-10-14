#include "gpu_macros.h"
#include "host_utils.h"
#include <cuda_fp16.h>
#include <iostream>
#include <cassert>

int main() {
    HostUtils utils;
    int sm61_gpu_idx = 2;  // 1080Ti, Pascal CUDA 6.1
    int sm75_gpu_idx = 0;  // 2080Ti, Turing CUDA 7.5
    int full_dim = 10000;
    int half_dim = full_dim;
    int half2_dim = full_dim / 2;


    // cuBLAS handles, constants, etc
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    BLAS_HANDLE_ERROR( cublasCreate( &handle ) );


    // Host data
    float* full_host;
    float* full_res;
    __half* half_host;
    __half* half_res;
    __half2* half2_host;
    __half2* half2_res;

    // Device data
    float* full_A;
    float* full_B;
    float* full_C;
    __half* half_A;
    __half* half_B;
    __half* half_C;
    __half2* half2_A;
    __half2* half2_B;
    __half2* half2_C;




    // Compute on sm_61 with full precision

    utils.recordStartTime();
    // Initialize full sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_res, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_A, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_B, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_C, full_dim * full_dim * sizeof(float) ) );

    for(int i = 0; i < full_dim * full_dim; ++i) {
        full_host[i] = 2;
    }

    HANDLE_ERROR( cudaMemcpy( full_A, full_host, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( full_B, full_host, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );


    // Compute full sm_61
    BLAS_HANDLE_ERROR( cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                full_dim,
                full_dim,
                full_dim,
                &alpha,
                full_A,
                full_dim,
                full_B,
                full_dim,
                &beta,
                full_C,
                full_dim
            ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( full_res, full_C, full_dim * full_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 32-bit SGEMM, " << full_dim << " elements, " << utils.timeDifference();
    std::cout << " seconds" << std::endl;

    for(int i = 0; i < full_dim * full_dim; ++i) {
        assert(full_res[i] == 4 * full_dim);
    }


    // Free sm_61 full
    HANDLE_ERROR( cudaFreeHost( full_host ) );
    HANDLE_ERROR( cudaFreeHost( full_res ) );
    HANDLE_ERROR( cudaFree( full_A ) );
    HANDLE_ERROR( cudaFree( full_B ) );
    HANDLE_ERROR( cudaFree( full_C ) );





//    for(int i = 0; i < full_dim * full_dim; ++i) {
//        half_precision_mat[i] = __float2half(full_precision_mat[i]);
//    }
//
//    for(int i = 0; i < half2_dim * half2_dim; ++i) {
//        half2_precision_mat[i] = __floats2half2_rn(full_precision_mat[2*i], full_precision_mat[(2*i)+1]);
//    }












    // Compute on sm_61 with __half

    // Compute on sm_61 with __half2


    // Compute on sm_75 with full precision

    // Compute on sm_75 with __half

    // Compute on sm_75 with __half2


    // cuBLAS free
    BLAS_HANDLE_ERROR( cublasDestroy( handle ) );
}
