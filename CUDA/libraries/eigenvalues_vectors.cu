/* 
 purpose:      Compute the eigenvalues and eigenvectors of a matrix A
               of N*N with N=5
 compile:      nvcc eigenvalues_vectors.cu -lcusolver
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusolverDn.h"


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\t", name, row+1, col+1, Areg);
        }
        printf("\n")
    }
}

int main(int argc, char **argv) 
{
    cusolverStatus_t cusolver_status;
    cusolverHandle_t cusolver_handle = NULL;
    cudaError_t cudaStat1 = cudaSuccess;

    int i, j;
    int N = 5;
    double  *A;
    
    // memory allocation
    A = (double *) malloc(N * N * sizeof(double));

    // set up matrix A[][] supposedly consisting of just eigenvectors

    A[0][0] = 1.96 ; A[1][0] = −6.49; A[2][0] = −0.47; A[3][0] = −7.20; A[4][0] = −0.65;
    A[1][0] = −6.49; A[1][1] = 3.80 ; A[2][1] = −6.39; A[3][1] = 1.50 ; A[4][1] = −6.34;
    A[2][0] = −0.47; A[1][2] = −6.39; A[2][2] = 4.17 ; A[3][2] = −1.51; A[4][2] = 2.67 ;
    A[3][0] = −7.20; A[1][3] = 1.50 ; A[2][3] = −1.51; A[3][3] = 5.70 ; A[4][3] = 1.80 ;
    A[4][0] = −0.65; A[1][4] = −6.34; A[2][4] = 2.67 ; A[3][4] = 1.80 ; A[4][4] = −7.10;

    // print out initial matrix content
    printf(" Matrix to be sent into DGEMM\n");
    printMatrix(N, N, A, N, "A")   

    // initiate the CUBLAS context
    cusolverDnCreate(&handle);

    cusolverDnDestroy(&handle);

    return(0);
}