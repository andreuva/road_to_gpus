/***************************************************************
*    2 LEVEL ATOM ATMOSPHERE SOLVER CUDA ROUTINES              *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
*    Compilation: nvcc -c routines_gpu.cu -o routines_gpu.o    *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_extension.h"

__global__ void kernel1() {
    printf("Hello APP from GPU!\n");
}

void launchGPU() { 
    kernel1<<<4,4>>>(); 
    cudaDeviceSynchronize();
}
