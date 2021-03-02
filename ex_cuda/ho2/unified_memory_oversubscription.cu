/* 
 * purpose:      testing/verifying CUDA Unified Memory for >=pascal 
 *               architectures; 
 *               n.b. the only question here is whether we are able to 
 *                    let the GPU work on data arrays much larger than what
 *                    is available on-board of the device, say with the 
 *                    8GB onboard memory of the gtx1080 master an array
 *                    of size 16GB allocated as cudaMallocManaged() on
 *                    the host
 *               n.b.2. the other real/unusual question is what could be a
 *                      reasonable one-dim array consuming 16GB ? and what
 *                      would be a tractable correctness test corresponding
 *                      to it ?
 *               result: yes, working great !
 *                       running 'watch nvidia-smi' in a 2nd terminal,
 *                       we really see the 8111MiB resident for a long time
 *                       even for times when the kernel is already back and 
 *                       we just wait for function check_array() to complete !
 * compilation:  nvcc ./unified_memory_oversubscription.cu
 * usage:        ./a.out
 */



#include <stdio.h>
#define DBLEARRAY16GB 2147483648




/* 
 * GPU kernel: working on managed unified memory
 * of really huge size, say 16GB, which is twice the
 * amount physically available on the device
 * n.b. threadIdx.x is of type unsigned int !
 *      so we should have access to 2^32 = 4294967296 
 *      different elements in terms of indices, hence
 *      sufficient to service the 2147483648 needed 
 *      within our huge-sized array but only if we 
 *      drop the sign !
 */
__global__ void KrnlDmmy(double *a)
{
    unsigned int i;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[i] = (double) i + 5.0e00;

    return;
}






/* 
 * host: array check
 * this is actually all but trivial in terms of achievable/required 
 * numerical accuracy ! theoretically for our regular array, a[0]=5, a[1]=6, 
 * a[2]=7 ... a[2147483647]=2147483652 with the requested number of 
 * DBLEARRAY16GB elements, we expect a total sum of 
 * (5+2147483652)*2147483648/2=2305843018877370368
 * which turns out to be really hard to compute :-)
 */
long double check_array(double *a)
{
    unsigned int i;
    long double rslt, prvs_rslt, dlta;

   /* 
    * receives the array a[] after it had been modified on the GPU 
    * and so we just want to run a sum over all its elements to 
    * check whether the GPU kernel had been working correctly; 
    */
    rslt = 0.0e00;
    for (i=0; i<DBLEARRAY16GB; i++) {
       prvs_rslt = rslt;
       rslt += a[i];
       dlta = rslt - prvs_rslt;
       if (dlta > (i + 5)) {
          printf("element %ld, of value %lf causes dlta %lf\n", i, a[i], dlta);
       }
    }

    return(rslt);
}





/* 
 * host: main  
 */
int main()
{
    long int huge;
    dim3 thrds_per_block, blcks_per_grid;
    double *c;
    long double sgnl;


   /* 
    * at first we want to call a simple kernel that writes something into
    * CUDA unified memory (to be host-allocated next); the specific 
    * challenge here is to figure out whether a 16GB array may still 
    * fit into the device despite its physical size being limited to 8GB !
    */
    huge = (long int) DBLEARRAY16GB * sizeof(double);
    printf("huge in Bytes: %ld sizeof(double): %d\n", huge, sizeof(double));
    c = NULL;
    cudaMallocManaged(&c, huge);
    thrds_per_block.x = 512;
    blcks_per_grid.x = ((long int) DBLEARRAY16GB) / thrds_per_block.x;
    printf("working with blcks_per_grid.x: %d\n", blcks_per_grid.x);
    KrnlDmmy<<<blcks_per_grid, thrds_per_block>>>(c);
    cudaDeviceSynchronize(); 
    printf("back from GPU kernel\n");

   /* 
    * if everything has gone well we should end up with 2305843018877370368
    * when adding together all content of all individual array elements, c[]
    */
    printf("c[0] %lf\n", c[0]);
    printf("c[9] %lf\n", c[9]);
    printf("c[99] %lf\n", c[99]);
    printf("c[999] %lf\n", c[999]);
    printf("c[9999] %lf\n", c[9999]);
    printf("c[99999] %lf\n", c[99999]);
    printf("c[999999] %lf\n", c[999999]);
    printf("c[9999999] %lf\n", c[9999999]);
    printf("c[99999999] %lf\n", c[99999999]);
    printf("c[999999999] %lf\n", c[999999999]);
    printf("c[2147483647] %lf\n", c[2147483647]);
    printf("c[DBLEARRAY16GB] %lf\n", c[DBLEARRAY16GB-1]);
    sgnl = check_array(c);
    printf("expecting 2305843018877370368 while receiving %llf\n", sgnl);


    cudaFree(c);

    return(0);
}
