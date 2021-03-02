#include <stdio.h>
#include <iostream>
#include <math.h>

__global__ void cuda_hello(){
    fprintf(stdout,"Hello World from GPU!\n");
}

int main() {
    fprintf(stdout,"Hello from the main program!!\n");
    cuda_hello<<<4,4>>>(); 
    return 0;
}