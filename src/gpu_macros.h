#ifndef NOCTURNAL_LLAMA_GPU_MACROS_H
#define NOCTURNAL_LLAMA_GPU_MACROS_H

#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <exception>


const char* cublasGetErrorString(cublasStatus_t status);
void cuda_error(cudaError_t e, const char *file, int code_line);
void cuBLAS_error(cublasStatus_t e, const char *file, int code_line);
void __cudaCheckError( const char *file, const int line );

#define HANDLE_ERROR(e) (cuda_error(e, __FILE__, __LINE__))
#define BLAS_HANDLE_ERROR(e) (cuBLAS_error(e, __FILE__, __LINE__))
#define CUDA_ERROR_CHECK
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

#endif // NOCTURNAL_LLAMA_GPU_MACROS_H