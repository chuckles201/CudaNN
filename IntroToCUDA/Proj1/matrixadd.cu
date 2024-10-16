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
#define matX 10000
#define matY 1000
#define MATOVR matX * matY
#define THREADS 256 // threads per-block
#define m1 3 // just for checking for now
#define m2 5

// defining dimensions for 3d vector
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

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

__global__ void addGPU3D(float *mat1, float *mat2, float *result, int nx, int ny, int nz) {
    // making sure we don't overstep overall dimension: has to do with how we initialized gridDims
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    int indy = blockIdx.y * blockDim.y + threadIdx.y;
    int indz = blockIdx.z * blockDim.z + threadIdx.z;

    if (indx<nx && indy<ny && indz < nz) {// if we didn't overstep bounds
        // add unrolled index!
        int ovr_index = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) // block overall amount
        * blockDim.x*blockDim.y*blockDim.z // converting to thread-amts
        + threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x*blockDim.y; // adding thread idx

        result[ovr_index] = mat1[ovr_index] + mat2[ovr_index];
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

    // initialize matrices for GPU
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);
    h_result = (float*)malloc(size);
    matrix_init(matsize,h_x,h_y,h_result); // changing memory locations

    // allocating,copying memory to/on GPU (don't cpy results)
    cudaMalloc(&d_x, size); // putting memory into GPU
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_result,size);
    cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice); // copying host to device
    cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);

    // for executing on GPU ***
    int blockSize = THREADS;
    int numBlocks = (matsize + blockSize -1) / blockSize;

    // Now creating functions for 3D GPU
    float *h_result_3d, *d_result_3d; // stores in memory
    int nx=100,ny=100,nz=1000; // 10m array
    dim3 blockSize3d(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
    dim3 numBlocks3d( // making sure we have enough dimesnions in 3d grid
        (nx + BLOCK_SIZE_X -1) / BLOCK_SIZE_X,
        (ny + BLOCK_SIZE_Y -1) / BLOCK_SIZE_Y,
        (nz + BLOCK_SIZE_Z -1) / BLOCK_SIZE_Z
    );
    cudaMalloc(&d_result_3d,size); // same size, just in 3d!
    

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
        addGPU3D<<<numBlocks3d,blockSize3d>>>(d_x,d_y,h_result_3d,nx,ny,nz);
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
    for (int i = 0; i<10; i++){
        double start = get_time();
        addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time2 = time;
    printf("GPU AVG TIME: |%f miliseconds|\n\n",time*1000.0);

    time = 0.0;

    // GPU run
    for (int i = 0; i<10; i++){
        double start = get_time();
        addGPU3D<<<numBlocks3d,blockSize3d>>>(d_x,d_y,h_result_3d,nx,ny,nz);
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time3 = time;

    printf("GPU3D AVG TIME: |%f miliseconds|\n\n",time*1000.0);



    // Testing GPU accuracy
    // creating memory for 3d result
    float *result_3d;
    result_3d = (float*)malloc(size);
    cudaMemcpy(h_result,d_result,size,cudaMemcpyDeviceToHost); // copying info back to host
    cudaMemcpy(h_result_3d,result_3d,size,cudaMemcpyDeviceToHost); // 3d to host

    bool correct = true;

    for (int i=0; i<matsize; i++) {
        if (h_result_cpu[i] != h_result_3d[i]) {
            correct = false;
            break;
        }
    }

    printf("All values accurate? --> |%s|\n\n",correct ? "correct":"incorrect");

    printf("GPU Speedup vs. CPU : |%fx|\n\n",time1/time2);
    printf("GPU 3D Speedup:%f\n",time2/time3);



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