// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>

// other libs
#include <stdio.h>
#include <iostream>

// global vars
#define matX 10000
#define matY 1000
#define MATOVR matX * matY
#define THREADS 256 // threads per-block
#define m1 3 // just for checking for now
#define m2 5


// defining Kernel
__global__ void addGPU(int size,float* mat1, float* mat2,float*result){
    //ovr. index
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) { // if inside bounds
        result[index] = mat1[index] + mat2[index];
    }
}

// defining CPU function
void addCPU(int size,float* mat1, float* mat2,float*result){

    for(int i = 0; i <size; i++) {
        result[i] = mat1[i] + mat2[i];
    }
}

// initializing memory for 2 matrices (on host)
void matrix_init(int size, float *mat1,float *mat2,float*mat3) { // pointer arrays (increment memory location)
    for (int i = 0; i<size; i++){ 
        mat1[i] = m1; // ith memory location in array
        mat2[i] = m2;
        mat3[i] = 0;
    }
}

int main(){
    

    // allocating memory (x&y are matrices)
    float *h_x,*h_y, *h_result; // memory for matices
    float *d_x, *d_y, *d_result; // memory on GPU

    int matsize = matX*matY; // size of unravelled matrix
    int size = matsize * sizeof(float); // memory

    // initialize matrices
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);
    h_result = (float*)malloc(size);
    matrix_init(matsize,h_x,h_y,h_result); // changing memory locations

    // allocating,copying memory to/on GPU
    cudaMalloc(&d_x, size); // putting memory into GPU
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_result,size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_result,h_result,size,cudaMemcpyHostToDevice);


    // executing on GPU ***
    int blockSize = THREADS;
    int numBlocks = (matsize + blockSize -1) / blockSize;
    addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
    cudaDeviceSynchronize();

    // copying results to CPU
    cudaMemcpy(h_x,d_x,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y,d_y,size,cudaMemcpyDeviceToHost);
    printf("GPU RESULTS: %f,%f,%f\n",h_result[0],h_result[9999],h_result[234232]);

    // allocating memory with ptrs to array
    float *h_x_cpu, *h_y_cpu, *h_result_cpu;
    h_x_cpu = (float*)malloc(size);
    h_y_cpu = (float*)malloc(size);
    h_result_cpu = (float*)malloc(size);

    // intializing CPU
    matrix_init(matsize,h_x_cpu,h_y_cpu,h_result_cpu); // initializing matrices
    addCPU(matsize,h_x_cpu,h_y_cpu,h_result_cpu);
    printf("CPU RESULTS : %f,%f,%f\n",h_result_cpu[0],h_result_cpu[234232],h_result_cpu[9999]);


    // 'warmup runs'


    // CPU, GPU benchmark implementations


    // Testing GPU accuracy
    
    // TODO: add error checking, timing, comparison

    // freeing memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_result);
    free(h_x_cpu);
    free(h_y_cpu);
    free(h_result_cpu);
    


    return 0;
}