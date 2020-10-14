# Testing cuBLAS GEMM Speeds

Using NVIDIA's 1080Ti and 2080Ti on Ubuntu 18.04 LTS, testing cuBLAS SGEMM, HGEMM, and GemmEx algorithms on 20,000 
element square matrices.  Examples of each are provided.

If you're here from Google, here's some information about NVIDIA's __half as defined in cuda_fp16.h:

  - The minimum and maximum numerical limits for __half are +/- 8,192.  I couldn't find this explicitly documented 
  anywhere else.  
  - __half2 cannot be used in the cuBLAS GEMM algorithms and is largely defined for custom CUDA kernels, from what I 
  can tell.


The following are the timings for a single iteration of each algorithm on each architecture:


CUDA 6.1 32-bit SGEMM, 400000000 elements per array, 1600 MB memory per array, 3.25769 sec

CUDA 6.1 16-bit HGEMM, 400000000 elements per array, 800 MB memory per array, 74.9325 sec

CUDA 6.1 16-32-bit GemmEx, 400000000 elements per array, 800 and 1600 MB memory for 16-/32-bit arrays, 2.39189 sec

CUDA 7.5 32-bit SGEMM, 400000000 elements per array, 1600 MB memory per array, 3.40309 sec

CUDA 7.5 16-bit HGEMM, 400000000 elements per array, 800 MB memory per array, 1.45896 sec

CUDA 7.5 16-32-bit GemmEx, 400000000 elements per array, 800 and 1600 MB memory for 16-/32-bit arrays, 1.40492 sec


Note that "Fast 16-bit float" compute is not available for the CUDA 6 compute level, except for the P100, which I am not
using here.  As a result, half precision exclusive GEMM is 20-25x slower than 32-bit or mixed 16-32-bit GEMM for the
1080Ti.  Also, even though the GemmEx mixed precision algorithm utilizes 32-bit intermediate data structures, the 2080Ti
 GemmEx outperforms the HGEMM, perhaps due to use of Tensor cores.