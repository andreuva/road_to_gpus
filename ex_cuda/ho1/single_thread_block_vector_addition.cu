/* 
 * purpose:      just a demo to show how vector addition can be done on 
 *               the GPU with just a single thread block
 * compilation:  nvcc ./single_thread_block_vector_addition.cu
 * usage:        ./a.out
 */ 

#include <stdio.h>

#define   N 100

/* 
 * GPU kernel 
 */
__global__ void VecAdd(float *A, float *B, float *C)
{
    int i;

    i = threadIdx.x;
    C[i] = A[i] + B[i];
}




/* 
 * host main  
 */
int main()
{
    int i;
    dim3 numBlocks, threadsPerBlock;
    float *A, *B, *C;

   /* 
    * using CUDA unified memory, first allocate
    * the memory then initialize with some dummy content        
    */
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    cudaMallocManaged(&C, N * sizeof(float));
    for (i = 0; i < N; i++) {
        A[i] = (float) i;
        B[i] = (float) (N - i);
        C[i] = (float) 0;
    }

   /* 
    * set up GPU kernel execution configuration 
    * however, this time we send in explicit parameters 
    * directly
    */
    threadsPerBlock.x = N;
    numBlocks.x = 1;
   
   /* launch the GPU kernel */
    VecAdd<<<1, N>>>(A, B, C);  
    cudaDeviceSynchronize();
 
   /* print result */
    for (i = 0; i < N; i++) {
        printf("%d %f\n", i, C[i]);
    }
   
   /* make clean */
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);


    return(0);
}
