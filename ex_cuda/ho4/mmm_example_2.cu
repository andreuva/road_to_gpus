/* 
 * purpose:      matrix-matrix multiplication using simplest matrices of 
 *               all squared dimension N x N; this is the simplest version
 *               disregarding all problems with memory access patterns  
 * result:       basically working;
 * compilation:  nvcc ./mmm_example_2.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define N 1600



/* 
 * GPU kernel for straightforward matrix-matrix multiplication
 * n.b. i,j are array/matrix indices starting from 0,1,2...
 */
__global__ void KrnlMMM(float **A, float **B, float **C)
{
    int i, j, k;
    float tmpC;

    i = ((blockIdx.x * blockDim.x) + threadIdx.x);
    j = ((blockIdx.y * blockDim.y) + threadIdx.y);
    tmpC = (float) 0;
    for (k=0; k<N; k++) {
        tmpC += A[i][k] * B[k][j]; 
    }
    C[i][j] = tmpC;

    return;
}




/* 
 * host main  
 */
int main()
{
    int i, j, k, i0, j0;
    dim3 threadsPerBlock, numBlocks;
    float **A, **B, **C, tC;


   /* 
    * using CUDA unified memory, first allocate 
    * the arrays in convenient 2D format, then 
    * initialize with some dummy content        
    */
    srand(time(0)); 
    cudaMallocManaged(&A, N * sizeof(float *));
    cudaMallocManaged(&B, N * sizeof(float *));
    cudaMallocManaged(&C, N * sizeof(float *));
    for (i = 0; i < N; i++) {
        cudaMallocManaged(&A[i], N * sizeof(float));
        cudaMallocManaged(&B[i], N * sizeof(float));
        cudaMallocManaged(&C[i], N * sizeof(float));
        for (j = 0; j < N; j++) {
            A[i][j] = (float) rand() / (float) RAND_MAX;
            B[i][j] = (float) 1  / A[i][j];
            C[i][j] = (float) 0;
        }
    }

   /* 
    * next we want to call a simple kernel that carries out 
    * matrix-matrix multiplication of A x B and stores the 
    * results into C; this is a far-from-optimal 1st approximation !
    */
    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    numBlocks.x = N / threadsPerBlock.x;
    numBlocks.y = N / threadsPerBlock.y;
    KrnlMMM<<<numBlocks, threadsPerBlock>>>(A, B, C);
    cudaDeviceSynchronize(); 

   /* 
    * just pick a random item and compute it explicitly for a check
    */
    i0 = (int) ((float) N * A[0][0]);
    j0 = (int) ((float) N * A[1][1]);
    printf("random check for C[%d][%d]\n", i0, j0);
    tC = (float) 0;
    for (k=0; k<N; k++) {
        tC += A[i0][k] * B[k][j0];
    }
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[i0][j0]);

    tC = (float) 0;
    for (k=0; k<N; k++) {
        tC += A[0][k] * B[k][0];
    }
    printf("check for C[0][0]\n");
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[0][0]);

   
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);


    return(0);
}
