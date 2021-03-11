/***************************************************************
*    2 LEVEL ATOM ATMOSPHERE SOLVER CUDA ROUTINES              *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
*    Compilation: NVCC                                         *
****************************************************************/
#include <cuda_runtime.h>
#include <cuda_extension.h>

__global__ void kernel1() { 
    printf("Hello APP from GPU!\n");
}

void launchGPU() { 
    kernel1<<<4,4>>>(); 
    cudaDeviceSynchronize();
}