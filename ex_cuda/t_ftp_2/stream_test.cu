/*
 *  usage:  nvcc ./stream_test.cu -o ./stream_legacy
 *          nvvp ./stream_legacy    ( or as root: 
 *                                    nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_legacy )
 *          ... versus ...
 *          nvcc --default-stream per-thread ./stream_test.cu -o ./stream_per-thread
 *          nvvp ./stream_per-thread    ( or as root: 
 *                                        nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_per-thread )
 */



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

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}

