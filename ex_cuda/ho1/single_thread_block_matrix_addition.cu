/* 
 * purpose:      just a demo to show how matrix addition can be done on 
 *               the GPU with just a single thread block, ie for rather 
 *               small sized underlying matrix dimensions
 * compilation:  nvcc ./single_thread_block_matrix_addition.cu
 * usage:        ./a.out
 */ 

#include <stdio.h>

#define   N 30

/* 
 * GPU kernel 
 */
__global__ void MatAdd(float **A, float **B, float **C)
{
    int i, j;

    i = threadIdx.x;
    j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
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
    threadsPerBlock.x = N;
    threadsPerBlock.y = N;
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
