/* 
 * purpose:      CUDA managed unified memory for >= pascal architectures;
 *               this version just uses cudaMallocManaged() on the host,
 *               then runs two kernels on the GPU to add together two arrays
 *               of size 1 GB and save the results into a third array;
 *               n.b. here we want to separate the initialization stage
 *                    from the actual calculation stage via introduction
 *                    of a second kernel  
 * result:       from profiling via 'nvprof ./a.out' we now see much
 *               more efficient computation once the initialization has
 *               been accomplished; interestingly the number of page faults
 *               has almost remained constant and so has the overall 
 *               execution time, so the separation into two kernels
 *               now provides very detailed insight into the cost of
 *               individual partial tasks;
 * compilation:  nvcc ./unified_memory_example_2.cu
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
    */
    thrds_per_block.x = 256;
    blcks_per_grid.x = ARRAYDIM / thrds_per_block.x;
    KrnlDmmyInit<<<blcks_per_grid, thrds_per_block>>>(a, b, c);
    cudaDeviceSynchronize(); 
    KrnlDmmyCalc<<<blcks_per_grid, thrds_per_block>>>(a, b, c);
    cudaDeviceSynchronize();
    //for (i=0; i<=100; i++) {
    //    printf("%6d%6.1f%6.1f%6.1f\n", i, a[i], b[i], c[i]);
    //}
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);


    return(0);
}
