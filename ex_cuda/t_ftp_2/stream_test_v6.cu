/*
 *  usage:  nvcc --default-stream per-thread -ccbin g++ -m64 -Xcompiler -fopenmp  ./stream_test_v6.cu -o ./stream_v6_per-thread  
 *          top   ( in a separate xterm )                                          
 *          ./stream_v6_per-thread     ( shows individual OpenMP threads in the 'top' view )
 *          nvvp ./stream_v6_per-thread    ( or as root: 
 *                                           nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_v6_per-thread )
 *
 *  purpose: essentially the same as ./stream_test_v5.cu however with an
 *           additional dummy loop (j, k) to make individual threads visible 
 *           for runtime monitoring within 'top'
 *  result:  nicely visible in the xterm, but here the non-perfect
 *           concurrency gets lost when considering the nvvp representation
 */

#include <stdio.h>
#include <omp.h>

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


    omp_set_num_threads(num_streams);
    #pragma omp parallel for 
    for (int i = 0; i < num_streams; i++) {
        int cpu_thread_id = omp_get_thread_num();
        printf("*** hello from thread %d \n", cpu_thread_id);
        cudaStreamCreate(&streams[i]);
        cudaMallocManaged(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);

        // just introduce a dummy task to also see 
        // individual threads working automously within top
        for (int j = 0; j < 20000; j++) {
            for (int k = 0; k < 500000; k++) {
            }
        }
    }

    // and a quick check of results because individual streams 
    // should have done identical calculations !
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        printf("*** %d %12.6lf%12.6lf%12.6lf\n", i, data[i][0], data[i][1], data[i][2]);
    }

    cudaDeviceReset();

    return 0;
}

