// cuda/ other libs
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// other libs
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>


// writing a function to examine how the SASS
// works. Testing vector loads/normal loads

// version 1: unrolled function
__global__ void vectorLoad(float *A,float *C) {
    // loading into cache...
    // then writing back to C
    __shared__ float Cs[4];
    Cs[0] = A[0];
    Cs[1] = A[1];
    Cs[2] = A[2];
    Cs[3] = A[3];

}

int main() {
    float *c_h,*a_h,*a_d,*c_d;
    a_h = (float*)malloc(sizeof(float)*4);
    c_h = (float*)malloc(sizeof(float)*4);
    for(int i = 0; i<4; i++){
        a_h[i] = 4;
    }

    // allocate memory in GPU
    cudaMalloc(&a_d,sizeof(float)*4);
    cudaMalloc(&c_d,sizeof(float)*4);

    cudaMemcpy(a_d,a_h,sizeof(float)*4,cudaMemcpyHostToDevice);
    vectorLoad<<<1,1>>>(a_d,c_d);
    cudaMemcpy(c_h,c_d,sizeof(float)*4,cudaMemcpyDeviceToHost);

    printf("C: %f,%f,%f,%f",c_h[0],c_h[1],c_h[2],c_h[3]);


    free(a_h);
    free(c_h);
    cudaFree(a_d);
    cudaFree(c_d);

    return 0;
}