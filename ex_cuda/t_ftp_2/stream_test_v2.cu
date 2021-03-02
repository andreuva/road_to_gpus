/*
 *  usage:  nvcc ./stream_test_v2.cu -o ./stream_v2_legacy
 *          nvvp ./stream_v2_legacy    ( or as root: 
 *                                       nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_v2_legacy )
 *          ... versus ...
 *          nvcc --default-stream per-thread ./stream_test_v2.cu -o ./stream_v2_per-thread
 *          nvvp ./stream_v2_per-thread    ( or as root: 
 *                                           nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_v2_per-thread )
 *
 *  purpose: just test whether substitution of cudaMalloc() with cudaMallocManaged() 
 *           will work
 *  result:  yes it does, however, one needs to be careful with synchronizing individual
 *           streams before accessing managed memory, hence the below inserted call was
 *           crucial, because without it only default stream 0 had printed out correct
 *           results and all other streams just 0 !
 *           n.b. out-commented line 'cudaMalloc(&data[i]...' would have worked together with 
 *                also commented lines 'cudaMemcpy(... printf(...' in terms of low level 
 *                checking of results
 *
 */

#include <stdio.h>


const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
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
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
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

