#include "gpu_macros.h"

const char* cublasGetErrorString(cublasStatus_t status)
{
	switch(status)
	{
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}


void cuda_error(cudaError_t e, const char *file, int code_line)
{
	if(e != cudaSuccess) {
		std::cerr << "CUDA execution error: " << cudaGetErrorString(e) << " (" << e;
		std::cerr << e << "), " << file << " at line " << code_line << std::endl;
		std::exit(EXIT_FAILURE);
	}
}


void cuBLAS_error(cublasStatus_t e, const char *file, int code_line)
{
	if(e != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cuBLAS execution error: " << cublasGetErrorString(e) << ", " << file <<  " at line " << code_line << std::endl;
		std::exit(EXIT_FAILURE);
	}
}


void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		         file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
	return;
}