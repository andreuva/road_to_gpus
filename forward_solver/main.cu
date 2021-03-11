/***************************************************************
*    2 LEVEL ATOM ATMOSPHERE SOLVER CUDA IMPLEMENTATION        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
*    Compilation: make                                         *
****************************************************************/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include "params.h"
#include "integratives.h"

int main(int argc, char **argv)
{
    const float a = 1;                          /* # dumping Voigt profile a=gam/(2^1/2*sig) */
    const float r = 1;                          /* # line strength XCI/XLI */
    const float eps = 1e-4;                     /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
    const float dep_col = 1;                    /* # Depolirarization colisions (delta) */
    const float Hd = 0.20;                      /* # Hanle depolarization factor [1/5, 1] */

    fprintf(stdout, "\n------------------- PARAMETERS OF THE PROBLEM ---------------------\n");
    fprintf(stdout, "optical thicknes of the lower boundary:            %1.1e \n", zl);
    fprintf(stdout, "optical thicknes of the upper boundary:            %1.1e \n", zu);
    fprintf(stdout, "resolution in the z axis:                          %1.3e \n", dz);
    fprintf(stdout, "total number of points in z:                       %i    \n", nz);
    fprintf(stdout, "lower/upper frequency limit :                      %1.3e   %1.3e \n", wl, wu);
    fprintf(stdout, "number points to sample the spectrum:              %i \n", nw);
    fprintf(stdout, "nodes in the gaussian quadrature (# dirs):         %i \n", qnd);
    /*fprintf(stdout, "T (isotermic) of the medium:                       %i \n", T);*/
    fprintf(stdout, "dumping Voigt profile:                             %1.3e \n", a);
    fprintf(stdout, "line strength XCI/XLI:                             %1.3e \n", r);
    fprintf(stdout, "Phot. dest. probability (LTE=1,NLTE=1e-4):         %1.3e \n", eps);
    fprintf(stdout, "Depolirarization colisions (delta):                %f \n", dep_col);
    fprintf(stdout, "Hanle depolarization factor [1/5, 1]:              %f \n", Hd);
    fprintf(stdout, "angular momentum of the levels (Ju, Jl):           (%i,%i) \n", ju, jl);
    fprintf(stdout, "Tolerance for finding the solution:                %f \n", tolerance);
    fprintf(stdout, "------------------------------------------------------------------\n\n");

    return 0;
}
