/* 
 * GPU kernel 
 */
__global__ void VecAdd(float *A, float *B, float *C)
{
    int i;

    i = threadIdx.x;
    C[i] = A[i] + B[i];
}



extern "C" void ntmdtr_(float *A, float *B, float *C, int *N)
{
    dim3 numBlocks, threadsPerBlock;
    float *AD, *BD, *CD;
    
   /* 
    * set up GPU kernel execution configuration 
    */
    threadsPerBlock.x = *N;
    numBlocks.x = 1;

   /* 
    * prepare device memory as we need to go the explicit
    * cudaMemcpy() way this time
    */
    cudaMalloc((void **) &AD, (*N) * sizeof(float));
    cudaMalloc((void **) &BD, (*N) * sizeof(float));
    cudaMalloc((void **) &CD, (*N) * sizeof(float));
    
    // transfer data to GPU 
    cudaMemcpy(AD, A, (*N) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(BD, B, (*N) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(CD, C, (*N) * sizeof(float), cudaMemcpyHostToDevice);

    // launch the GPU kernel 
    VecAdd<<<numBlocks, threadsPerBlock>>>(AD, BD, CD);  
    cudaDeviceSynchronize();

    // copy back the result from the GPU, A and B should be unchanged !
    cudaMemcpy(C, CD, (*N) * sizeof(float), cudaMemcpyDeviceToHost);

    // free memory on the GPU
    cudaFree(AD);  
    cudaFree(BD);  
    cudaFree(CD);  

    return;
}
