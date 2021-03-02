#include <stdio.h>
#include <iostream>
#include <math.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<4,4>>>(); 
    return 0;
}