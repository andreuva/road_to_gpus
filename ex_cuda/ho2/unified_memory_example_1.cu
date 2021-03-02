/* 
 * purpose:      CUDA managed unified memory for >= pascal architectures;
 *               this version just uses cudaMallocManaged() on the host,
 *               then runs a kernel on the GPU to add together two arrays
 *               of size 1 GB and save the results into a third array;
 * result:       working great ! when running a loop over 25 attemps with
 *               'watch nvidia-smi' open in a background terminal, we see 
 *               Memory-Usage  3185MiB /  8114MiB 
 *               n.b. for visual clarity, the printout section below should be 
 *                    commented out when starting to do some profiling runs,
 *                    e.g. nvprof ./a.out
 * compilation:  nvcc ./unified_memory_example_1.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define ARRAYDIM 268435456



/* 
 * GPU kernel working with unified memory which had been 
 * allocated using cudaMallocManaged() on the host 
 */
__global__ void KrnlDmmy(float *x, float *y, float *z)
{
    int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    x[i] = (float) i;
    y[i] = (float) (i + 1);
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
    * next we want to call a simple kernel that sets array elements
    * a[] and b[] with thread-specific values and then adds together
    * these values and stores back the result into array c[]
    */
    thrds_per_block.x = 256;
    blcks_per_grid.x = ARRAYDIM / thrds_per_block.x;
    KrnlDmmy<<<blcks_per_grid, thrds_per_block>>>(a, b, c);
    cudaDeviceSynchronize(); 
    //for (i=0; i<=100; i++) {
    //    printf("%6d%6.1f%6.1f%6.1f\n", i, a[i], b[i], c[i]);
    //}
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);


    return(0);
}
