/* 
 * purpose:      matrix-matrix multiplication using simple square matrices of 
 *               dimension N x N; this is the shared memory variant where 
 *               a particular thread-block (BS x BS) operates on a sequence of 
 *               pairs of submatrices also dimensioned BS x BS in a 
 *               dot-product-like fashion to form a particular sub-block 
 *               in the result matrix; the block indices of the block-grid
 *               may serve to navigate through the matrices and identify
 *               individual submatrices;
 *               n.b. 1600 % 16 = N % BS = 0
 * result:       much faster than without using shared memory, ie approx 10x
 * compilation:  nvcc ./mmm_example_3.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define N 1600
#define BS  16



/* 
 * GPU kernel for straightforward matrix-matrix multiplication
 * using shared memory
 * n.b. i,j are array/matrix indices starting from 0,1,2...
 *      at first sight it may be confusing that i goes with threadIdx.y
 *      while j corresponds to threadIdx.x
 */
__global__ void KrnlMMM(float **A, float **B, float **C)
{
    int i, j, k, Blckk;
    __shared__ float BlckA[BS][BS], BlckB[BS][BS], BlckC[BS][BS];
    float tmpC;


   /*
    * target block to compute here starts at element 
    * C[blockIdx.y*BS][blockIdx.x*BS] and will involve 
    * all combinations of blocks starting with 
    * A[blockIdx.y*BS][0BS,1BS,2BS...] 
    * and
    * B[0BS,1BS,2BS...][blockIdx.x*BS]
    * where the latter series of A/B blocks will be combined
    * in a dot-product-like fashion, so that eventually we 
    * will end up with a number of N / BS such block-products;
    * at first we shall initialize our resulting block, which
    * is easily achieved by letting all threads operate on their
    * target item in parallel;
    */
    i = threadIdx.y;
    j = threadIdx.x;
    BlckC[i][j] = (float) 0;
    for (Blckk = 0; Blckk < (N/BS); Blckk++) {

       /*
        * get sub-matrices into shared memory; this will also be
        * easy since we are working with a threadblock of size
        * BS x BS, hence each thread just needs to copy a single 
        * element from global into shared memory
        */
        BlckA[i][j] = A[(blockIdx.y*BS)+i][(Blckk*BS)+j];
        BlckB[i][j] = B[(Blckk*BS)+i][(blockIdx.x*BS)+j];
        __syncthreads();
     
       /*
        * next the currently selected pair of BlckA[][] 
        * and BlckB[][] shall be multiplied together such
        * that each individual thread i,j determines its own
        * target element inside BlckC[][] 
        */
        tmpC = (float) 0;
        for (k = 0; k<BS; k++) {
            tmpC += BlckA[i][k] * BlckB[k][j]; 
        }
        BlckC[i][j] += tmpC;
        __syncthreads();
       
    }
   
   /*
    * and the only thing remaining is to copy back the resulting submatrix
    * BlckC[][] into global memory to a particular sector of matrix 
    * C[][]; again, this can be done in parallel by all threads 
    * each of them copying just a single element
    */
    C[(blockIdx.y*BS)+i][(blockIdx.x*BS)+j] = BlckC[i][j];
    __syncthreads();


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
    * matrix-matrix multiplication of A x B and stores the results 
    * into C; this is an improved version based on shared memory
    */
    threadsPerBlock.x = BS;
    threadsPerBlock.y = BS;
    numBlocks.x = N / threadsPerBlock.x;
    numBlocks.y = N / threadsPerBlock.y;
    KrnlMMM<<<numBlocks, threadsPerBlock>>>(A, B, C);
    cudaDeviceSynchronize(); 

   /* 
    * just pick a random item and compute it explicitly for a check
    */
    i0 = (int) ((float) N * A[0][0]);
    j0 = (int) ((float) N * A[1][1]);
    tC = (float) 0;
    for (k=0; k<N; k++) {
        tC += A[i0][k] * B[k][j0];
    }
    printf("checking C[%d][%d]\n", i0, j0);
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[i0][j0]);

    tC = (float) 0;
    for (k=0; k<N; k++) {
        tC += A[0][k] * B[k][0];
    }
    printf("checking C[0][0]\n");
    printf("explicit calc %6.3f\n", tC);
    printf("kernel calc   %6.3f\n", C[0][0]);

   
    cudaFree(C);
    cudaFree(B);
    cudaFree(A);


    return(0);
}
