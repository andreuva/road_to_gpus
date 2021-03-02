/* 
 * purpose:      just a demo to show how matrix addition can be done on 
 *               the GPU with just a single thread block, ie for rather 
 *               small sized underlying matrix dimensions
 *               n.b. here we want to consider threadblock dimensions
 *                    different from the actual shape of the arrays
 * compilation:  nvcc ./single_thread_block_matrix_addition_v2.cu
 * usage:        ./a.out
 */ 

#include <stdio.h>

#define   N 64

/* 
 * GPU kernel 
 */
__global__ void MatAdd(float **A, float **B, float **C)
{
    int i, j, block;

    block = blockIdx;
    i = threadIdx.x;
    j = threadIdx.y;
    if ( (i < N) && (j < N)) {
       C[i][j] = A[i][j] + B[i][j];
    }
    printf("process (%i,%i) from block i% finished\n", i,j,block)
}




/* 
 * host main  
 */
int main()
{
    int i, j;
    dim3 threadsPerBlock, numBlocks;
    float **A, **B, **C;

   /* 
    * using CUDA unified memory, first allocate 
    * the memory in convenient 2D format, then 
    * initialize with some dummy content        
    */
    cudaMallocManaged(&A, N * sizeof(float *));
    cudaMallocManaged(&B, N * sizeof(float *));
    cudaMallocManaged(&C, N * sizeof(float *));
    for (i = 0; i < N; i++) {
        cudaMallocManaged(&A[i], N * sizeof(float));
        cudaMallocManaged(&B[i], N * sizeof(float));
        cudaMallocManaged(&C[i], N * sizeof(float));
        for (j = 0; j < N; j++) {
            A[i][j] = (float) ((i * N) + j);
            B[i][j] = (N * N) - A[i][j];
            C[i][j] = (float) 0;
        }
    }

   /* set up GPU kernel execution configuration */
    threadsPerBlock.x = N + 1;
    threadsPerBlock.y = N + 1;
    numBlocks.x = 1;
   
   /* launch the GPU kernel */
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);  
    cudaDeviceSynchronize();
 
   /* print result */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d %d %f\n", i, j, C[i][j]);
        }
    }
   
   /* make clean */
    for (i = 0; i < N; i++) {
        cudaFree(C[i]);
        cudaFree(B[i]);
        cudaFree(A[i]);
    }
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);


    return(0);
}
