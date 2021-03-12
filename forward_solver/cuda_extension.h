#ifndef __CUDAEXT__
#define __CUDAEXT__
extern "C"{
    #include "params.h"
    #include <complex.h>

    static const short nw = (wu-wl)/dw + 1;                /* # points to sample the spectrum */
    static const short nz = (zu-zl)/dz + 1;                /* # number of points in the z axes */

    extern void psi_calc(double deltaum[], double deltaup[], \
                 double psim[], double psio[], double psip[], int mode);
    extern void RTE_SC_solve(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);
    void launchGPU(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);
}
#endif
