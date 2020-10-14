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
    float alpha = 1.0f;
    float beta = 0.0f;


    // Host data
    float* full_host1;
    float* full_res1;
    float* full_host2;
    float* full_res2;
    __half* half_host1;
    __half* half_res1;
    __half* half_host2;
    __half* half_res2;
    __half2* half2_host1;
    __half2* half2_res1;
    __half2* half2_host2;
    __half2* half2_res2;

    // Device data
    float* full_A1;
    float* full_B1;
    float* full_C1;
    float* full_A2;
    float* full_B2;
    float* full_C2;
    __half* half_A1;
    __half* half_B1;
    __half* half_C1;
    __half* half_A2;
    __half* half_B2;
    __half* half_C2;
    __half2* half2_A1;
    __half2* half2_B1;
    __half2* half2_C1;
    __half2* half2_A2;
    __half2* half2_B2;
    __half2* half2_C2;


    // Compute on sm_61 with full precision

    utils.recordStartTime();
    // Initialize full sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    cublasHandle_t full_handle1;
    BLAS_HANDLE_ERROR( cublasCreate( &full_handle1 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host1, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_res1, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_A1, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_B1, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_C1, full_dim * full_dim * sizeof(float) ) );

    for(int i = 0; i < full_dim * full_dim; ++i) {
        full_host1[i] = 2;
    }

    HANDLE_ERROR( cudaMemcpy( full_A1, full_host1, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( full_B1, full_host1, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute full sm_61
    BLAS_HANDLE_ERROR( cublasSgemm(
                full_handle1,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                full_dim,
                full_dim,
                full_dim,
                &alpha,
                full_A1,
                full_dim,
                full_B1,
                full_dim,
                &beta,
                full_C1,
                full_dim
            ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( full_res1, full_C1, full_dim * full_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 32-bit SGEMM, " << full_dim * full_dim << " elements per array, ";
    std::cout << full_dim * full_dim * sizeof(float) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " seconds" << std::endl;

    for(int i = 0; i < full_dim * full_dim; ++i) {
        assert(full_res1[i] == 4 * full_dim);
    }


    // Free sm_61 full
    BLAS_HANDLE_ERROR( cublasDestroy( full_handle1 ) );
    HANDLE_ERROR( cudaFreeHost( full_host1 ) );
    HANDLE_ERROR( cudaFreeHost( full_res1 ) );
    HANDLE_ERROR( cudaFree( full_A1 ) );
    HANDLE_ERROR( cudaFree( full_B1 ) );
    HANDLE_ERROR( cudaFree( full_C1 ) );



// Compute on sm_61 with half precision

    utils.recordStartTime();
    // Initialize full sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    cublasHandle_t half_handle1;
    BLAS_HANDLE_ERROR( cublasCreate( &half_handle1 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_host1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_res1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_A1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_B1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_C1, half_dim * half_dim * sizeof(__half) ) );

    for(int i = 0; i < half_dim * half_dim; ++i) {
        half_host1[i] = 2;
    }

    HANDLE_ERROR( cudaMemcpy( half_A1, half_host1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( half_B1, half_host1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute full sm_61
    BLAS_HANDLE_ERROR( cublasHgemm(
            half_handle1,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            half_dim,
            half_dim,
            half_dim,
            &alpha,
            half_A1,
            half_dim,
            half_B1,
            half_dim,
            &beta,
            half_C1,
            half_dim
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( half_res1, half_C1, half_dim * half_dim * sizeof(__half), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 16-bit HGEMM, " << half_dim * half_dim << " elements per array, ";
    std::cout << half_dim * half_dim * sizeof(__half) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " seconds" << std::endl;

    for(int i = 0; i < half_dim * half_dim; ++i) {
        assert(half_res1[i] == 4 * half_dim);
    }


    // Free sm_61 full
    BLAS_HANDLE_ERROR( cublasDestroy( half_handle1 ) );
    HANDLE_ERROR( cudaFreeHost( half_host1 ) );
    HANDLE_ERROR( cudaFreeHost( half_res1 ) );
    HANDLE_ERROR( cudaFree( half_A1 ) );
    HANDLE_ERROR( cudaFree( half_B1 ) );
    HANDLE_ERROR( cudaFree( half_C1 ) );

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



}
