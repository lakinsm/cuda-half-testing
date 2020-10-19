#include "gpu_macros.h"
#include "host_utils.h"
#include <cuda_fp16.h>
#include <iostream>
#include <cassert>


int main() {
    __half temp1 = __float2half(1.0f);
    __half temp2 = __float2half(5.0f);
    __half temp3 = temp1 - temp2;
    std::cout << temp3 << std::endl;





    HostUtils utils;
    int sm61_gpu_idx = 2;  // 1080Ti, Pascal CUDA 6.1
    int sm75_gpu_idx = 0;  // 2080Ti, Turing CUDA 7.5 with Tensor cores
    int full_dim = 20000;
    int half_dim = full_dim;

    utils.recordStartTime();
    float temp4 = 100.0f;
    for(int i = 0; i < 100000; ++i) {
        float temp5 = std::log(temp4);
    }
    utils.recordStopTime();
    std::cout << utils.timeDifference() << std::endl;

    utils.recordStartTime();
    __half temp6 = __float2half(100.0f);
    for(int i = 0; i < 100000; ++i) {
        __half temp7 = __float2half(std::log(__half2float(temp6)));
    }
    utils.recordStopTime();
    std::cout << utils.timeDifference() << std::endl;


    // cuBLAS handles, constants, etc
    float full_alpha = 1.0f;
    float full_beta = 0.0f;
    __half half_alpha = __float2half(1.0f);
    __half half_beta = __float2half(0.0f);


    // Host data
    float* full_host_A1;
    float* full_host_B1;
    float* full_res1;
    float* full_host_A2;
    float* full_host_B2;
    float* full_res2;
    __half* half_host_A1;
    __half* half_host_B1;
    __half* half_res1;
    __half* half_host_A2;
    __half* half_host_B2;
    __half* half_res2;
    __half* mixed_host_A1;
    __half* mixed_host_B1;
    float* mixed_res1;
    __half* mixed_host_A2;
    __half* mixed_host_B2;
    float* mixed_res2;

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
    __half* mixed_A1;
    __half* mixed_B1;
    float* mixed_C1;
    __half* mixed_A2;
    __half* mixed_B2;
    float* mixed_C2;


    // Compute on sm_61 with full precision

    utils.recordStartTime();
    // Initialize full sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    cublasHandle_t full_handle1;
    BLAS_HANDLE_ERROR( cublasCreate( &full_handle1 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host_A1, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host_B1, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_res1, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_A1, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_B1, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_C1, full_dim * full_dim * sizeof(float) ) );

    for(int i = 0; i < full_dim * full_dim; ++i) {
        full_host_A1[i] = 2;
        full_host_B1[i] = -2;
    }

    HANDLE_ERROR( cudaMemcpy( full_A1, full_host_A1, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( full_B1, full_host_B1, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute full sm_61
    BLAS_HANDLE_ERROR( cublasSgemm(
                full_handle1,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                full_dim,
                full_dim,
                full_dim,
                &full_alpha,
                full_A1,
                full_dim,
                full_B1,
                full_dim,
                &full_beta,
                full_C1,
                full_dim
            ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( full_res1, full_C1, full_dim * full_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 32-bit SGEMM, " << full_dim * full_dim << " elements per array, ";
    std::cout << full_dim * full_dim * sizeof(float) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    for(int i = 0; i < full_dim * full_dim; ++i) {
        assert(full_res1[i] == -4 * full_dim);
    }


    // Free sm_61 full
    BLAS_HANDLE_ERROR( cublasDestroy( full_handle1 ) );
    HANDLE_ERROR( cudaFreeHost( full_host_A1 ) );
    HANDLE_ERROR( cudaFreeHost( full_host_B1 ) );
    HANDLE_ERROR( cudaFreeHost( full_res1 ) );
    HANDLE_ERROR( cudaFree( full_A1 ) );
    HANDLE_ERROR( cudaFree( full_B1 ) );
    HANDLE_ERROR( cudaFree( full_C1 ) );



    // Compute on sm_61 with half precision

    utils.recordStartTime();
    // Initialize half sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    cublasHandle_t half_handle1;
    BLAS_HANDLE_ERROR( cublasCreate( &half_handle1 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_host_A1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_host_B1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_res1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_A1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_B1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_C1, half_dim * half_dim * sizeof(__half) ) );

    __half val1 = __float2half(2.0f);
    __half nval1 = __float2half(-2.0f);
    for(int i = 0; i < half_dim * half_dim; ++i) {
        half_host_A1[i] = val1;
        half_host_B1[i] = nval1;
    }

    HANDLE_ERROR( cudaMemcpy( half_A1, half_host_A1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( half_B1, half_host_B1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute half sm_61
    BLAS_HANDLE_ERROR( cublasHgemm(
            half_handle1,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            half_dim,
            half_dim,
            half_dim,
            &half_alpha,
            half_A1,
            half_dim,
            half_B1,
            half_dim,
            &half_beta,
            half_C1,
            half_dim
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( half_res1, half_C1, half_dim * half_dim * sizeof(__half), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 16-bit HGEMM, " << half_dim * half_dim << " elements per array, ";
    std::cout << half_dim * half_dim * sizeof(__half) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    for(int i = 0; i < half_dim * half_dim; ++i) {
        assert(__half2float(half_res1[i]) == -8192);
    }


    // Free sm_61 half
    BLAS_HANDLE_ERROR( cublasDestroy( half_handle1 ) );
    HANDLE_ERROR( cudaFreeHost( half_host_A1 ) );
    HANDLE_ERROR( cudaFreeHost( half_host_B1 ) );
    HANDLE_ERROR( cudaFreeHost( half_res1 ) );
    HANDLE_ERROR( cudaFree( half_A1 ) );
    HANDLE_ERROR( cudaFree( half_B1 ) );
    HANDLE_ERROR( cudaFree( half_C1 ) );




    // Compute on sm_61 with mixed precision

    utils.recordStartTime();
    // Initialize mixed sm_61
    HANDLE_ERROR( cudaSetDevice( sm61_gpu_idx ) );
    cublasHandle_t mixed_handle1;
    BLAS_HANDLE_ERROR( cublasCreate( &mixed_handle1 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_host_A1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_host_B1, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_res1, half_dim * half_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_A1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_B1, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_C1, half_dim * half_dim * sizeof(float) ) );

    __half mixed_val1 = __float2half(2.0f);
    __half mixed_nval1 = __float2half(-2.0f);
    for(int i = 0; i < half_dim * half_dim; ++i) {
        mixed_host_A1[i] = mixed_val1;
        mixed_host_B1[i] = mixed_nval1;
    }

    HANDLE_ERROR( cudaMemcpy( mixed_A1, mixed_host_A1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( mixed_B1, mixed_host_B1, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute mixed sm_61
    BLAS_HANDLE_ERROR( cublasGemmEx(
            mixed_handle1,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            half_dim,
            half_dim,
            half_dim,
            &full_alpha,
            mixed_A1,
            CUDA_R_16F,
            half_dim,
            mixed_B1,
            CUDA_R_16F,
            half_dim,
            &full_beta,
            mixed_C1,
            CUDA_R_32F,
            half_dim,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( mixed_res1, mixed_C1, half_dim * half_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 6.1 16-32-bit GemmEx, " << half_dim * half_dim << " elements per array, ";
    std::cout << half_dim * half_dim * sizeof(__half) / 1000000 << " and ";
    std::cout << half_dim * half_dim * sizeof(float) / 1000000 << " MB memory for 16-/32-bit arrays, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    for(int i = 0; i < half_dim * half_dim; ++i) {
        assert(mixed_res1[i] == -4 * half_dim);
    }


    // Free sm_61 mixed
    BLAS_HANDLE_ERROR( cublasDestroy( mixed_handle1 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_host_A1 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_host_B1 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_res1 ) );
    HANDLE_ERROR( cudaFree( mixed_A1 ) );
    HANDLE_ERROR( cudaFree( mixed_B1 ) );
    HANDLE_ERROR( cudaFree( mixed_C1 ) );





    // Compute on sm_75 with full precision

    utils.recordStartTime();
    // Initialize full sm_75
    HANDLE_ERROR( cudaSetDevice( sm75_gpu_idx ) );
    cublasHandle_t full_handle2;
    BLAS_HANDLE_ERROR( cublasCreate( &full_handle2 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host_A2, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_host_B2, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&full_res2, full_dim * full_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_A2, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_B2, full_dim * full_dim * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&full_C2, full_dim * full_dim * sizeof(float) ) );

    for(int i = 0; i < full_dim * full_dim; ++i) {
        full_host_A2[i] = 2;
        full_host_B2[i] = -2;
    }

    HANDLE_ERROR( cudaMemcpy( full_A2, full_host_A2, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( full_B2, full_host_B2, full_dim * full_dim * sizeof(float), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute full sm_75
    BLAS_HANDLE_ERROR( cublasSgemm(
            full_handle2,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            full_dim,
            full_dim,
            full_dim,
            &full_alpha,
            full_A2,
            full_dim,
            full_B2,
            full_dim,
            &full_beta,
            full_C2,
            full_dim
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( full_res2, full_C2, full_dim * full_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 7.5 32-bit SGEMM, " << full_dim * full_dim << " elements per array, ";
    std::cout << full_dim * full_dim * sizeof(float) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    for(int i = 0; i < full_dim * full_dim; ++i) {
        assert(full_res2[i] == -4 * full_dim);
    }


    // Free sm_75 full
    BLAS_HANDLE_ERROR( cublasDestroy( full_handle2 ) );
    HANDLE_ERROR( cudaFreeHost( full_host_A2 ) );
    HANDLE_ERROR( cudaFreeHost( full_host_B2 ) );
    HANDLE_ERROR( cudaFreeHost( full_res2 ) );
    HANDLE_ERROR( cudaFree( full_A2 ) );
    HANDLE_ERROR( cudaFree( full_B2 ) );
    HANDLE_ERROR( cudaFree( full_C2 ) );



    // Compute on sm_75 with half precision

    utils.recordStartTime();
    // Initialize full sm_75
    HANDLE_ERROR( cudaSetDevice( sm75_gpu_idx ) );
    cublasHandle_t half_handle2;
    BLAS_HANDLE_ERROR( cublasCreate( &half_handle2 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_host_A2, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_host_B2, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&half_res2, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_A2, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_B2, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&half_C2, half_dim * half_dim * sizeof(__half) ) );

    __half val2 = __float2half(2.0f);
    __half nval2 = __float2half(-2.0f);
    for(int i = 0; i < half_dim * half_dim; ++i) {
        half_host_A2[i] = val2;
        half_host_B2[i] = nval2;
    }

    HANDLE_ERROR( cudaMemcpy( half_A2, half_host_A2, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( half_B2, half_host_B2, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute half sm_75
    BLAS_HANDLE_ERROR( cublasHgemm(
            half_handle2,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            half_dim,
            half_dim,
            half_dim,
            &half_alpha,
            half_A2,
            half_dim,
            half_B2,
            half_dim,
            &half_beta,
            half_C2,
            half_dim
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( half_res2, half_C2, half_dim * half_dim * sizeof(__half), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 7.5 16-bit HGEMM, " << half_dim * half_dim << " elements per array, ";
    std::cout << half_dim * half_dim * sizeof(__half) / 1000000 << " MB memory per array, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    // CUDA __half has numerical limits (minimum and maximum values) of +/- 8192
    for(int i = 0; i < half_dim * half_dim; ++i) {
        assert(__half2float(half_res2[i]) == -8192);
    }


    // Free sm_75 half
    BLAS_HANDLE_ERROR( cublasDestroy( half_handle2 ) );
    HANDLE_ERROR( cudaFreeHost( half_host_A2 ) );
    HANDLE_ERROR( cudaFreeHost( half_host_B2 ) );
    HANDLE_ERROR( cudaFreeHost( half_res2 ) );
    HANDLE_ERROR( cudaFree( half_A2 ) );
    HANDLE_ERROR( cudaFree( half_B2 ) );
    HANDLE_ERROR( cudaFree( half_C2 ) );




    // Compute on sm_75 with mixed precision

    utils.recordStartTime();
    // Initialize mixed sm_75
    HANDLE_ERROR( cudaSetDevice( sm75_gpu_idx ) );
    cublasHandle_t mixed_handle2;
    BLAS_HANDLE_ERROR( cublasCreate( &mixed_handle2 ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_host_A2, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_host_B2, half_dim * half_dim * sizeof(__half), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&mixed_res2, half_dim * half_dim * sizeof(float), cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_A2, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_B2, half_dim * half_dim * sizeof(__half) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&mixed_C2, half_dim * half_dim * sizeof(float) ) );

    __half mixed_val2 = __float2half(2.0f);
    __half mixed_nval2 = __float2half(-2.0f);
    for(int i = 0; i < half_dim * half_dim; ++i) {
        mixed_host_A2[i] = mixed_val2;
        mixed_host_B2[i] = mixed_nval2;
    }

    HANDLE_ERROR( cudaMemcpy( mixed_A2, mixed_host_A2, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( mixed_B2, mixed_host_B2, half_dim * half_dim * sizeof(__half), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );


    // Compute mixed sm_75
    BLAS_HANDLE_ERROR( cublasGemmEx(
            mixed_handle2,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            half_dim,
            half_dim,
            half_dim,
            &full_alpha,
            mixed_A2,
            CUDA_R_16F,
            half_dim,
            mixed_B2,
            CUDA_R_16F,
            half_dim,
            &full_beta,
            mixed_C2,
            CUDA_R_32F,
            half_dim,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    HANDLE_ERROR( cudaMemcpy( mixed_res2, mixed_C2, half_dim * half_dim * sizeof(float), cudaMemcpyDeviceToHost ) );
    utils.recordStopTime();

    std::cout << std::endl;
    std::cout << "CUDA 7.5 16-32-bit GemmEx, " << half_dim * half_dim << " elements per array, ";
    std::cout << half_dim * half_dim * sizeof(__half) / 1000000 << " and ";
    std::cout << half_dim * half_dim * sizeof(float) / 1000000 << " MB memory for 16-/32-bit arrays, ";
    std::cout << utils.timeDifference() << " sec" << std::endl;

    for(int i = 0; i < half_dim * half_dim; ++i) {
        assert(mixed_res2[i] == -4 * half_dim);
    }


    // Free sm_75 mixed
    BLAS_HANDLE_ERROR( cublasDestroy( mixed_handle2 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_host_A2 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_host_B2 ) );
    HANDLE_ERROR( cudaFreeHost( mixed_res2 ) );
    HANDLE_ERROR( cudaFree( mixed_A2 ) );
    HANDLE_ERROR( cudaFree( mixed_B2 ) );
    HANDLE_ERROR( cudaFree( mixed_C2 ) );
}
