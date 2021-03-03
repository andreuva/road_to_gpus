/* 
 * purpose:      just a simple check whether a matrix, A, is composed 
 *               of eigenvectors only, in which case A^t x A = E 
 *               hence the inverse, A^-1, is simply the transpose, A^t,
 *               resulting in the unit matrix, E, from the above matrix
 *               matrix multiplication;
 *               n.b. here we want to make use of CUBLAS on the GPU 
 * compile:      nvcc chck_ev.cu -lcublas
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


int main(int argc, char **argv) 
{
   int N, i, j;
   double alpha, beta, *A, **A2D, *Adev, *B, **B2D, *Bdev, *C, **C2D, *Cdev;
   cublasStatus_t stat;
   cublasHandle_t handle;
   cublasOperation_t Atype, Btype;


   // memory allocation and parameter set up
   N = 5;
   alpha = (double) 1;
   beta = (double) 0;
   Atype = CUBLAS_OP_N;
   Btype = CUBLAS_OP_N;
  
   A = (double *) malloc(N * N * sizeof(double));
   B = (double *) malloc(N * N * sizeof(double));
   C = (double *) malloc(N * N * sizeof(double));
   A2D = (double **) malloc(N * sizeof(double *));
   B2D = (double **) malloc(N * sizeof(double *));
   C2D = (double **) malloc(N * sizeof(double *));
   for (i = 0; i < N; i++) {
       A2D[i] = (double *) malloc(N * sizeof(double));
       B2D[i] = (double *) malloc(N * sizeof(double));
       C2D[i] = (double *) malloc(N * sizeof(double));
   }

   // set up matrix A2D[][] supposedly consisting of just eigenvectors
   A2D[0][0] =  0.30; A2D[0][1] = -0.61; A2D[0][2] =  0.40; A2D[0][3] =  0.37; A2D[0][4] = -0.49;
   A2D[1][0] =  0.51; A2D[1][1] = -0.29; A2D[1][2] = -0.41; A2D[1][3] =  0.36; A2D[1][4] =  0.61;
   A2D[2][0] =  0.08; A2D[2][1] = -0.38; A2D[2][2] = -0.66; A2D[2][3] = -0.50; A2D[2][4] = -0.40;
   A2D[3][0] =  0.00; A2D[3][1] = -0.45; A2D[3][2] =  0.46; A2D[3][3] = -0.62; A2D[3][4] =  0.46;
   A2D[4][0] =  0.80; A2D[4][1] =  0.45; A2D[4][2] =  0.17; A2D[4][3] = -0.31; A2D[4][4] = -0.16;

   // get the inverse of A2D[][] from simply the transpose (if really just eigenvectors)
   for (i = 0; i < N; i++) {
       for (j = 0; j < N; j++) {
           B2D[i][j] = A2D[j][i];
       }
   }

   // print out initial matrix content
   printf(" Matrix to be sent into DGEMM\n");
   for (i = 0; i < N; i++) {
       for (j = 0; j < N; j++) {
           printf("%10.2lf", A2D[i][j]); 
       }
       printf("\n");
   }      

   // copy content of A2D[][] and B2D[][] into their linear versions A[] and B[] --- column wise !
   for (i = 0; i < N; i++) {
       for (j = 0; j < N; j++) {
           A[(i*N)+j] = A2D[j][i];
           B[(i*N)+j] = B2D[j][i];
       }
   }

   // initiate the CUBLAS context
   cublasCreate(&handle);

   // allocate device memory
   cudaMalloc(&Adev, N * N * sizeof(double));
   cudaMalloc(&Bdev, N * N * sizeof(double));
   cudaMalloc(&Cdev, N * N * sizeof(double));

   // copy contents of arrays A[], B[] into device memory
   cublasSetMatrix(N, N, sizeof(double), &A[0], N, Adev, N);
   cublasSetMatrix(N, N, sizeof(double), &B[0], N, Bdev, N);

   // call CUBLAS routine cublasDgemm on the GPU, note the order Bdev[] x Adev[]
   stat = cublasDgemm(handle, Atype, Btype, N, N, N, &alpha, Bdev, N, Adev, N, &beta, Cdev, N);
   if ( stat != CUBLAS_STATUS_SUCCESS ) {
      printf("CUBLAS error \n");
      exit(99);
   }
  
   // retrive content of matrix matrix product from device memory
   cublasGetMatrix(N, N, sizeof(double), Cdev, N, &C[0], N);

   // print out results, hence the unit matrix if the assumption above was correct 
   printf(" Matrix matrix product\n");
   for (i = 0; i < N; i++) {
       for (j = 0; j < N; j++) {
           C2D[i][j] = C[(j*N)+i];
           printf("%10.2lf", C2D[i][j]);
       }
       printf("\n");
   }

   // and free up allocated memory and destroy the CUBLAS context
   cudaFree(Cdev);
   cudaFree(Bdev);
   cudaFree(Adev);
   cublasDestroy(handle);
   for (i = N-1; i >= 0; i--) {
       free(C2D[i]);
       free(B2D[i]);
       free(A2D[i]);
   }
   free(C2D);
   free(B2D);
   free(A2D);
   free(C);
   free(B);
   free(A);



   return(0);
}
