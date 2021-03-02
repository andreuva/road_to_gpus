/*
 *  usage:  nvcc --default-stream per-thread ./stream_test_v4.cu -o ./stream_v4_per-thread
 *          nvvp ./stream_v4_per-thread    ( or as root: 
 *                                           nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_v4_per-thread )
 *
 *  purpose: modify the kernel code to really use a block grid and 
 *           see what happens if grids are called in individual streams
 *
 *  result:  well, back to a quasi-sequential pattern instead of a concurrent 
 *           one with individual streams running in parallel; 
 *           so, since the first kernel launch of stream 0 had already occupied
 *           all available SM cores, all subsequent streams had to wait for resources
 *           to become vacant again !
 *           n.b. here the dummy call to the default stream needed to be commented out !
 *
 */

#include <stdio.h>


const int N = 1 << 20;

__global__ void kernel(float *x)
{
    int i;
    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    x[i] = sqrt(pow(3.14159, (int) threadIdx.x));
}

int main()
{
    const int num_streams = 8;
    float localx[N];

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        //cudaMalloc(&data[i], N * sizeof(float));
        cudaMallocManaged(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<N/64, 64, 0, streams[i]>>>(data[i]);

        // launch a dummy kernel on the default stream
        // kernel<<<1, 1>>>(0);
    }

    // and a quick check of results because individual streams 
    // should have done identical calculations !
    for (int i = 0; i < num_streams; i++) {
        // cudaMemcpy(localx, data[i], N * sizeof(float), cudaMemcpyDeviceToHost);
        // printf("*** %d %12.6lf%12.6lf%12.6lf\n", i, localx[0], localx[1], localx[2]);
        cudaStreamSynchronize(streams[i]);
        printf("*** %d %12.6lf%12.6lf%12.6lf\n", i, data[i][0], data[i][1], data[i][2]);
    }

    cudaDeviceReset();

    return 0;
}

