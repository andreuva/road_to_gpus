#ifndef __CUDAEXT__ 
#define __CUDAEXT__ 
extern "C"{
   void launchGPU(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);
}
#endif