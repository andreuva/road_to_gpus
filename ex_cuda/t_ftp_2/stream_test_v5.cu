/*
 *  usage:  nvcc --default-stream per-thread -ccbin g++ -m64 -Xcompiler -fopenmp  ./stream_test_v5.cu -o ./stream_v5_per-thread  
 *          nvvp ./stream_v5_per-thread    ( or as root: 
 *                                           nvvp -vm /usr/lib64/jvm/jre-1.8.0/bin/java ./stream_v5_per-thread )
 *
 *
 *  purpose: just check whether combining OpenMP with CUDA streams will work 
 *           in a straightforward way   
 *  result:  yes, in principle ! however concurrency of streams becomes 
 *           visible only within nvvp not within nvidia-smi !
 *           there is also a certain delay for concurrent streams, hence
 *           going parallel via OpenMP is not without overhead !
 *           but essentially, this could be the simplest way to fully exhaust 
 *           all available CPU and GPU resources on a particular machine;
 *           n.b. if we launch 8 OpenMP threads and have got only
 *                6 physical cores on the CPU we might see that in
 *                the nvvp profile too !
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

