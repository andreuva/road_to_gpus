/***************************************************************
        DOT PRODUCT FUNCTION IMPLEMENTED FOR THE CPU
    AUTHOR: ANDRES VICENTE AREVALO      DATE:10/03/2021
****************************************************************/

// Function that compute the dot product between two arrays
float dotprod(float *A, float *B, int len){
    int i;
    float result = 0;

    for (i = 0; i < len; i++){
        result += A[i] * B[i];
    }

    return result;
}