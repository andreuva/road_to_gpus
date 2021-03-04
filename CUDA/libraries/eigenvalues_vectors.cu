/* 
 purpose:      Compute the eigenvalues and eigenvectors of a matrix A
               of N*N with N=5
 compile:      nvcc eigenvalues_vectors.cu -lcusolver
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>


void printMatrixArray(int m, int n, const double *A, int lda, const char* name){
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\t", name, row+1, col+1, Areg);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrix(int m, int n, double **A, int lda, const char* name){
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row][col];
            printf("%s(%d,%d) = %f\t", name, row+1, col+1, Areg);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char **argv){
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusolverDnHandle_t cusolver_handle = NULL;
    cudaError_t cudaStat = cudaSuccess;

    cusolverEigMode_t jobz;
    cublasFillMode_t uplo;

    int N, i, j;
    double *A, **A_host, *Evectors, *Evalues;
    double *device_A = NULL;
    double *device_Eigenvalues = NULL;
    int *device_info = NULL;
    int info_gpu

    // memory allocation and parameter set up
    N = 5;

    A = (double *) malloc(N * N * sizeof(double));
    Evectors = (double *) malloc(N * N * sizeof(double));
    Evalues = (double *) malloc(N * sizeof(double));
    A_host = (double **) malloc(N * sizeof(double *));
    for (i = 0; i < N; i++) {
        A_host[i] = (double *) malloc(N * sizeof(double));
    }

    // set up matrix A_host[][] to compute the eigenvalues
    A_host[0][0] =  1.96; A_host[0][1] = -6.49; A_host[0][2] = -0.47; A_host[0][3] = -7.20; A_host[0][4] = -0.65;
    A_host[1][0] = -6.49; A_host[1][1] =  3.80; A_host[1][2] = -6.39; A_host[1][3] =  1.50; A_host[1][4] = -6.34;
    A_host[2][0] = -0.47; A_host[2][1] = -6.39; A_host[2][2] =  4.17; A_host[2][3] = -1.51; A_host[2][4] =  2.67;
    A_host[3][0] = -7.20; A_host[3][1] =  1.50; A_host[3][2] = -1.51; A_host[3][3] =  5.70; A_host[3][4] =  1.80;
    A_host[4][0] = -0.65; A_host[4][1] = -6.34; A_host[4][2] =  2.67; A_host[4][3] =  1.80; A_host[4][4] = -7.10;

    // copy content of A_host[][] into their linear version A[] --- column wise !
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[(i*N)+j] = A_host[i][j];
        }
    }

    printf("================================================")
    printMatrix(N, N, A_host, N, "A_host");
    printMatrixArray(N, N, A, N, "A");
    printf("================================================")

    jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    uplo = CUBLAS_FILL_MODE_LOWER; 

    // Step 1: initiate the cusolverDn context
    cusolver_status = cusolverDnCreate(&cusolver_handle);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // step 2: copy matrices to device
    cudaStat = cudaMalloc ((void**)&device_A, sizeof(double) * N * N);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc ((void**)&device_Eigenvalues, sizeof(double) * N);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMalloc ((void**)&device_info, sizeof(int));
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(device_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);


    // step 5: retrieve the results from device memory
    cudaStat = cudaMemcpy(Evalues, device_Eigenvalues, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(Evectors, device_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(&info_gpu, device_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);

    // step 6: print out results, ie eigenvalues and corresponding eigenvectors
    printf(" Eigenvalues\n");
    printMatrixArray(1, N, Evalues, 1, "lambda");
    printf("================================================")
    printf(" Corresponding Eigenvectors\n");
    printMatrixArray(N, N, Evectors, N, "V");
    printf("================================================")
    printf("\n");
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%10.2lf", Evectors[(j*m)+i]);
        }
        printf("\n");
    }
  
    // step 7: free all allocated memory and destroy context
   cudaFree(device_A);
   cudaFree(device_Eigenvalues);
   cudaFree(device_info);
   
   cusolverDnDestroy(cusolver_handle);
   cudaDeviceReset();

    for (i = N-1; i >= 0; i--) {
        free(A_host[i]);
    }
    free(A_host);
    free(A);

    return(0);
}
