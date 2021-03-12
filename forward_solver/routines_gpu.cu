/***************************************************************
*    2 LEVEL ATOM ATMOSPHERE SOLVER CUDA ROUTINES              *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
*    Compilation: nvcc -c routines_gpu.cu -o routines_gpu.o    *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "params.h"
extern "C"{
    #include "routines_cpu.c"
}
#include "cuda_extension.h"

__global__ void kernel1() {
    printf("Hello APP from GPU!\n");
}

void launchGPU(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]) { 
    kernel1<<<4,4>>>(); 
    cudaDeviceSynchronize();
}
