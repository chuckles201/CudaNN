// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>

// other libs
#include <stdio.h>
#include <iostream>

// global vars
# define matrixDim = {100,100} // 100*100 matrices

// TODO : finish matrix addition kernel with 2d threads and 1d block!

// defining Kernel
__global__ void add(float* mat1, float* mat2){
    ;
}

// initializing memory for 2 matrices (on host)
void matrix_init(int size, float *mat1,float *mat2) { // pointer arrays (increment memory location)
    for (int i = 0; i<size; i++){
        mat1[i] = 0; // changing given memory location
        mat2[i] = 0;
    }
}

int main(){

    // defining block&thread dimensions (arbitrary #)
    


    // freeing memory


    return 0;
}