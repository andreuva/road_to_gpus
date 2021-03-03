/* 
 * purpose:      CUDA managed unified memory for >= pascal architectures;
 *               this version just uses the variant of globally managed 
 *               memory hopefully set aside on the device, but for the rest 
 *               everything remains pretty similar to previous attempts;
 * result:       from profiling via 'nvprof ./a.out' we now see pretty
 *               good results too, practically identical to the prefetching 
 *               run which showed best performance so far; no page faults
 *               since the globally __managed__ resides on the GPU anyway
 * compilation:  nvcc ./unified_memory_example_5.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define ARRAYDIM 268435456

/* 
 * managed variable declaration for GPU memory (total of 3GB)
 */
__device__ __managed__ float x[ARRAYDIM], y[ARRAYDIM], z[ARRAYDIM];




/* 
 * GPU kernel doing the initialization
 */
__global__ void KrnlDmmyInit()
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
__global__ void KrnlDmmyCalc()
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

   /* 
    * so all we want to do is calling simple kernels that 
    * (i) initialize array elements a[] and b[] with thread-specific 
    * values and 
    * (ii) add together these values and store back the results into 
    * array c[] where the latter task shall be repeated within a loop 
    * over 100 iterations
    */
    thrds_per_block.x = 256;
    blcks_per_grid.x = ARRAYDIM / thrds_per_block.x;
    KrnlDmmyInit<<<blcks_per_grid, thrds_per_block>>>();
    cudaDeviceSynchronize(); 
    //printf("initialization completed\n");
    //printf("x[10]   %f y[10]   %f z[10]   %f\n", x[10],   y[10],   z[10]);
    //printf("x[100]  %f y[100]  %f z[100]  %f\n", x[100],  y[100],  z[100]);
    //printf("x[1000] %f y[1000] %f z[1000] %f\n", x[1000], y[1000], z[1000]);
    for (i=0; i<100; i++) {
        KrnlDmmyCalc<<<blcks_per_grid, thrds_per_block>>>();
        cudaDeviceSynchronize();
        //printf("iteration %d\n", i);
    }


    return(0);
}
