/* 
 * purpose:      just a simple example of using CUSOLVER to obtain all
 *               eigenvalues and corresponding eigenvectors
 *               for a 5 x 5 symmetric matrix
 * ref:          https://docs.nvidia.com/cuda/cusolver/index.html#eig_examples
 * compile:      nvcc chck_cusolver_syevd.cu -lcudart -lcusolver
 * result:       Eigenvalues
 *                  -11.07     -6.23      0.86      8.87     16.09
 *               Corresponding Eigenvectors
 *                   -0.30     -0.61     -0.40     -0.37      0.49
 *                   -0.51     -0.29      0.41     -0.36     -0.61
 *                   -0.08     -0.38      0.66      0.50      0.40
 *                   -0.00     -0.45     -0.46      0.62     -0.46
 *                   -0.80      0.45     -0.17      0.31      0.16
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>


int main(int argc, char **argv) 
{
   int i, j, lwork, info_gpu, *devInfo;
   double *d_A, *d_W, *d_work;
   cusolverDnHandle_t cusolverH;
   cusolverStatus_t cusolver_status;
   cusolverEigMode_t jobz;
   cublasFillMode_t uplo;
   cudaError_t cudaStat;
   const int m = 5;
   const int lda = m;
   double W[m]; 
   double V[lda*m];
   double A[lda*m] = { 1.96, -6.49, -0.47, -7.20, -0.65, 
                      -6.49,  3.80, -6.39,  1.50, -6.34,
                      -0.47, -6.39,  4.17, -1.51,  2.67,
                      -7.20,  1.50, -1.51,  5.70,  1.80,
                      -0.65, -6.34,  2.67,  1.80, -7.10};
   jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
   uplo = CUBLAS_FILL_MODE_LOWER; 


   // print out initial matrix content
   printf(" Matrix to be sent into cusolverDnDsyevd\n");
   for (i = 0; i < lda; i++) {
       for (j = 0; j < m; j++) {
           printf("%10.2lf", A[(i*lda)+j]);
       }
       printf("\n");
   }


   // step 1: create cusolver/cublas handle
   cusolver_status = cusolverDnCreate(&cusolverH);
   assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

   // step 2: copy matrices to device
   cudaStat = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
   assert(cudaSuccess == cudaStat);
   cudaStat = cudaMalloc ((void**)&d_W, sizeof(double) * m);
   assert(cudaSuccess == cudaStat);
   cudaStat = cudaMalloc ((void**)&devInfo, sizeof(int));
   assert(cudaSuccess == cudaStat);
   cudaStat = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
   assert(cudaSuccess == cudaStat);

   // step 3: query working space of syevd
   cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);
   assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
   cudaStat = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
   assert(cudaSuccess == cudaStat);

   // step 4: compute eigenvalues/eigenvectors
   cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);
   assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
   cudaStat = cudaDeviceSynchronize();
   assert(cudaSuccess == cudaStat);

   // step 5: retrieve the results from device memory
   cudaStat = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
   assert(cudaSuccess == cudaStat);
   cudaStat = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
   assert(cudaSuccess == cudaStat);
   cudaStat = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
   assert(cudaSuccess == cudaStat);

   // step 6: print out results, ie eigenvalues and corresponding eigenvectors
   printf(" Eigenvalues\n");
   for (i = 0; i < lda; i++) {
       printf("%10.2lf", W[i]);
   }
   printf("\n");
   printf(" Corresponding Eigenvectors\n");
   for (i = 0; i < lda; i++) {
       for (j = 0; j < m; j++) {
           printf("%10.2lf", V[(j*m)+i]);
       }
       printf("\n");
   }

   // step 7: free all allocated memory
   cudaFree(d_A);
   cudaFree(d_W);
   cudaFree(devInfo);
   cudaFree(d_work);
   cusolverDnDestroy(cusolverH);
   cudaDeviceReset();


   return(0);
}
