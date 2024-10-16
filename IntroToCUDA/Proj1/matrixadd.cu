// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>

// other libs
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// global vars
#define matX 100
#define matY 1000
#define MATOVR matX * matY
#define THREADS 256 // threads per-block
#define m1 3 // just for checking for now
#define m2 5

// function for timing --> ignore error for clock_monotonic
double get_time() {
    // creating holder for time
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC,&ts); // store at ts
    return ts.tv_sec + ts.tv_nsec * 1e-9; // nanoseconds and seconds


}

// defining Kernel
__global__ void addGPU(int size,float* mat1, float* mat2,float*result){
    //ovr. index
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) { // if inside bounds
        result[index] = mat1[index] + mat2[index];
    }
}

// defining CPU function
void addCPU(int matsize,float* mat1, float* mat2,float*result){

    for(int i = 0; i < matsize; i++) {
        result[i] = mat1[i] + mat2[i];
    }
}

// initializing memory for 2 matrices (on host)
void matrix_init(int size, float *mat1,float *mat2,float*mat3) { // pointer arrays (increment memory location)
    for (int i = 0; i<size; i++){ 
        mat1[i] = (float)m1; // ith memory location in array
        mat2[i] = (float)m2;
        mat3[i] = (float)0;
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
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice); // copying host to device
    cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_result,h_result,size,cudaMemcpyHostToDevice);


    // for executing on GPU ***
    int blockSize = THREADS;
    int numBlocks = (matsize + blockSize -1) / blockSize;

    // allocating memory with ptrs to array
    float *h_x_cpu, *h_y_cpu, *h_result_cpu;
    h_x_cpu = (float*)malloc(size);
    h_y_cpu = (float*)malloc(size);
    h_result_cpu = (float*)malloc(size);

    // intializing CPU
    matrix_init(matsize,h_x_cpu,h_y_cpu,h_result_cpu); // initializing matrices


    // 'warmup runs'
    for (int i=0; i<3; i++){
        addCPU(matsize,h_x_cpu,h_y_cpu,h_result_cpu);
        addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
        cudaDeviceSynchronize();
    }

    // CPU, GPU1D, GPU3D benchmark implementations
    // CPU run
    double time = 0;
    for (int i = 0; i<10; i++){
        double start = get_time();
        addCPU(matsize,h_x_cpu,h_y_cpu,h_result_cpu);
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time1 = time;

    printf("\n\nCPU AVG TIME: |%f miliseconds|\n",(float)time*1000);
    time = 0.0;

    // GPU run
    time = 0;
    for (int i = 0; i<10; i++){
        double start = get_time();
        addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time2 = time;

    printf("GPU AVG TIME: |%f miliseconds|\n\n",time*1000.0);



    // Testing GPU accuracy
    cudaMemcpy(h_result,d_result,size,cudaMemcpyDeviceToHost); // copying info back to host
    bool correct = true;

    for (int i=0; i<matsize; i++) {
        if (h_result[i] != h_result_cpu[i]) {
            correct = false;
            break;
        }
    }
    printf("All values accurate? --> |%s|\n\n",correct ? "correct":"incorrect");

    printf("GPU Speedup vs. CPU : |%fx|\n\n",time1/time2);



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