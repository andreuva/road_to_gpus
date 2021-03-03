/* 
 * purpose:      CUDA managed unified memory for >= pascal architectures;
 *               this version just uses cudaMallocManaged() on the host,
 *               then runs kernels on the GPU to add together two arrays
 *               of size 1 GB and save the results into a third array;
 *               n.b. here we want to again separate the initialization 
 *                    stage but then run the actual calculation in a
 *                    loop over 100 iterations to get a little better
 *                    statistics from the profiler
 * result:       from profiling via 'nvprof ./a.out' we now see greatly
 *               improved compute performance and much better memory
 *               bandwidth at again almost identical page fault statistics
 * compilation:  nvcc ./unified_memory_example_3.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define ARRAYDIM 268435456



/* 
 * GPU kernel doing the initialization
 */
__global__ void KrnlDmmyInit(float *x, float *y, float *z)
{
    int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    x[i] = (float) i;
    y[i] = (float) (i + 1);

    return;
}



/* 
 * GPU kernel doing the calculation, ie adding together two arrays
 */
__global__ void KrnlDmmyCalc(float *x, float *y, float *z)
{
    int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    z[i] = x[i] + y[i];

    return;
}







/* 
 * host main  
 */
int main()
{
    int i, cudaRtrn;
    dim3 thrds_per_block, blcks_per_grid;
    float *a, *b, *c;

   /* 
    * Let us make use of cudaMallocManaged() to allocate 3 arrays 
    * of size 1 GB each for subsequent usage on the GPU. 
    */
    if (cudaRtrn = cudaMallocManaged(&a, ARRAYDIM * sizeof(float)) != 0) {
       printf("*** allocation failed for array a[], %d ***\n", cudaRtrn);
    }
    if (cudaRtrn = cudaMallocManaged(&b, ARRAYDIM * sizeof(float)) != 0) {
       printf("*** allocation failed for array b[], %d ***\n", cudaRtrn);
    }
    if (cudaRtrn = cudaMallocManaged(&c, ARRAYDIM * sizeof(float)) != 0) {
       printf("*** allocation failed for array c[], %d ***\n", cudaRtrn);
    }

   /* 
    * next we want to call simple kernels that (i) initialize array 
    * elements a[] and b[] with thread-specific values and (ii) add 
    * together these values and store back the results into array c[]
    * the latter within a loop over 100 iterations
    */
    thrds_per_block.x = 256;
    blcks_per_grid.x = ARRAYDIM / thrds_per_block.x;
    KrnlDmmyInit<<<blcks_per_grid, thrds_per_block>>>(a, b, c);
    cudaDeviceSynchronize(); 
    for (i=0; i<100; i++) {
        KrnlDmmyCalc<<<blcks_per_grid, thrds_per_block>>>(a, b, c);
        cudaDeviceSynchronize();
    }
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);


    return(0);
}
