// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>

// other libs
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// global descriptors
#define MATM1 100
#define MATN1 100

#define MATM2 100
#define MATN2 100

//////////////////////////////////////////////////////////////////
// helper functions
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_nsec * 1e-9 + ts.tv_sec;

}

void matrixInit(int size, float *matrix) {
    for (int i = 0; i<size; i++) {
        matrix[i] = 10.0;
    }
}
/////////////////////////////////////////////////////////////////////
// CPU matmul
void matMulCPU(float *A, float *B, float *C, int mn, int m1, int n2) {
    for (int i = 0; i<m1; i++){ // each collumn of new matrix
        for (int j =0; j<n2; j++) { // each row in given collumn
            float dot = 0; // dot product at given point
            for (int k = 0; k < mn; k++) {
                dot += A[i*mn + k] * B[k*n2+i];
            }
            C[i*m1 + j] = dot;
            printf("ID:%d,(%d,%d) : %f\n", i*n2 + j,i,j,dot);
        }


    }
}
/////////////////////////////////////////////////////////////////////

int main() {

    // simple CPU implementation 
    float *mat1, *mat2, *result_cpu;
    int matsize1 = MATM1 * MATN1;
    int matsize2 = MATM2 * MATN2;
    int matsize3 = MATM1 * MATN2;
    mat1 = (float*)malloc(sizeof(float)*matsize1);
    mat2 = (float*)malloc(sizeof(float)*matsize2);
    result_cpu = (float*)malloc(sizeof(float)*matsize3);
    matrixInit(matsize1,mat1); // assinging values to memory positions
    matrixInit(matsize2,mat2);

    matMulCPU(mat1,mat2,result_cpu,MATN1,MATM1,MATN2);

    printf("Checking... %f,%f,%f", result_cpu[100],result_cpu[90],result_cpu[0]);

    // freeing memory
    free(mat1);
    free(mat2);
    free(result_cpu);





}