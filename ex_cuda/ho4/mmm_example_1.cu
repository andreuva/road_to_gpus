/* 
 * purpose:      matrix-matrix multiplication using simplest matrices of 
 *               all squared dimension N x N; this is the simplest version
 *               disregarding all problems with memory access patterns; 
 * result:       basically working;
 * compilation:  nvcc ./mmm_example_1.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define N 1600



/* 
 * GPU kernel for straightforward matrix-matrix multiplication
 * notice the decomposition of a single built-in index into
 * matrix column/row indices i,j
 */
__global__ void KrnlMMM(float *A, float *B, float *C)
{
    int i, j, k;
    float tmpC;

    i = (((blockIdx.x * blockDim.x) + threadIdx.x) / N) + 1;
    j = (((blockIdx.x * blockDim.x) + threadIdx.x) % N) + 1;
    tmpC = (float) 0;
    for (k=1; k<=N; k++) {
        tmpC += A[((i-1)*N)+k-1] * B[((k-1)*N)+j-1]; 
    }
    C[((i-1)*N)+j-1] = tmpC;

    return;
}




/* 
 * host main  
 */
int main()
{
    int i, cudaRtrn, i0, i1, j1, k;
    dim3 threadsPerBlock, numBlocks;
    float *A, *B, *C, tC;

   /* 
    * Let us make use of cudaMallocManaged() to allocate 3 
    * 1D arrays representing matrices of dimension N x N
    */
    if (cudaRtrn = cudaMallocManaged(&A, N * N * sizeof(float)) != 0) {
       printf("*** allocation failed for array A[], %d ***\n", cudaRtrn);
    }
    if (cudaRtrn = cudaMallocManaged(&B, N * N * sizeof(float)) != 0) {
       printf("*** allocation failed for array B[], %d ***\n", cudaRtrn);
    }
    if (cudaRtrn = cudaMallocManaged(&C, N * N * sizeof(float)) != 0) {
       printf("*** allocation failed for array C[], %d ***\n", cudaRtrn);
    }


   /* 
    * fill arrays with dummy random numbers
    */
    srand(time(0)); 
    for (i=0; i<N*N; i++) {
        A[i] = (float) rand() / (float) RAND_MAX;                        
        B[i] = (float) 1 / A[i];
    }


   /* 
    * next we want to call a simple kernel that carries out 
    * matrix-matrix multiplication of A x B and stores the 
    * results into C; this is a far-from-optimal 1st approximation !
    */
    threadsPerBlock.x = 256;
    numBlocks.x = (N * N) / threadsPerBlock.x;             
    KrnlMMM<<<numBlocks, threadsPerBlock>>>(A, B, C);
    cudaDeviceSynchronize(); 


   /* 
    * just pick a random item and compute it explicitly for a check
    */
    i0 = (int) ((float) (N * N) * A[0]);
    i1 = (i0 / N) + 1;
    j1 = (i0 % N) + 1;
    printf("random check for C[%d][%d]\n", i1, j1);
    tC = (float) 0;
    for (k=1; k<=N; k++) {
        tC += A[((i1-1)*N)+k-1] * B[((k-1)*N)+j1-1];
    }
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[i0]);

    tC = (float) 0;
    for (k=0; k<N; k++) {
        tC += A[k] * B[k*N];
    }
    printf("check for C[0][0]\n");
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[0]);

   
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);


    return(0);
}
