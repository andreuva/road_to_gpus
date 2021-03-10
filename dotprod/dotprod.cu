/***************************************************************
        DOT PRODUCT FOR NVIDIA GPUS AND CPU CHECK
    AUTHOR: ANDRES VICENTE AREVALO      DATE:10/03/2021
****************************************************************/

/* Host code that uses dotprod_cpu.c */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

// Define the dimension of the array
#define ARRAYDIM 32768

extern "C"
float dotprod(float *A, float *B, int len);

// shared memory managed for the GPU
__device__ __managed__ float A[ARRAYDIM], B[ARRAYDIM], result;


__global__ void array_init(){    
    
    int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;

    A[i] = 2;
    B[i] = 1;

    return;
}


// Function of the dotproduct in the kernel
__global__ void dotprod_kernel(){    
    int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;

    result += A[i] * B[i];

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Program main to do the dot product and compare it to C
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    float gpu_result, cpu_result;
    int devID = findCudaDevice(argc, (const char **)argv);
    unsigned int num_threads = 1024, num_blocks;

    num_blocks = ARRAYDIM/num_threads;

    dim3 grid(num_blocks,1,1);
    dim3 threads(num_threads,1,1);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    array_init<<< grid, threads >>>();
    cudaDeviceSyncronize();
    dotprod_kernel<<< grid, threads >>>();
    cudaDeviceSyncronize();

    sdkStopTimer(&timer);
    printf("Processing time for GPU code: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    gpu_result = result;
    result = 0;

    cpu_result = dotprod(A, B, ARRAYDIM);

    printf("CPU result is: %f\nGPU result is: %f", cpu_result, gpu_result)

}