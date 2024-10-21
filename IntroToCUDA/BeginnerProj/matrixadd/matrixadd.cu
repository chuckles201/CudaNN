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

// defining dimensions for 3d vector
#define BLOCK_SIZE_X 8
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
        int ovr_idx = indx + indy*nx + indz*nx*ny;
        
        printf("*");
        printf("GOOD-ID:%d, (%d,%d,%d)\n",ovr_idx,indx,indy,indz);
        result[ovr_idx] = mat1[ovr_idx] + mat2[ovr_idx];
        
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

    ///////////////////////////////////////////////////////
    // Now creating functions for 3D GPU
    float *h_result_3d, *d_result_3d; // stores in memory
    int nx=100,ny=100,nz=10; // 10m array
    dim3 blockSize3d(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
    dim3 numBlocks3d( // making sure we have enough dimesnions in 3d grid
        (nx + blockSize3d.x -1) / BLOCK_SIZE_X,
        (ny + blockSize3d.y -1) / BLOCK_SIZE_Y,
        (nz + blockSize3d.z -1) / BLOCK_SIZE_Z
    );
    h_result_3d = (float*)malloc(size);
    cudaMalloc(&d_result_3d,size); // same size, just in 3d!
    //////////////////////////////////////////////////////////

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
        cudaDeviceSynchronize();
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time2 = time;
    printf("GPU AVG TIME: |%f miliseconds|\n\n",time*1000.0);

    time = 0.0;

    // GPU3D run
    for (int i = 0; i<10; i++){
        double start = get_time();
        addGPU3D<<<numBlocks3d,blockSize3d>>>(d_x,d_y,d_result_3d,nx,ny,nz);
        cudaDeviceSynchronize();
        double end = get_time();
        time += (end-start);

    }
    time = time/10;
    double time3 = time;

    printf("GPU3D AVG TIME: |%f miliseconds|\n\n",time*1000.0);



    // Testing GPU accuracy
    // creating memory for 3d result
    cudaMemcpy(h_result,d_result,size,cudaMemcpyDeviceToHost); // copying info back to host
    cudaMemcpy(h_result_3d,d_result_3d,size,cudaMemcpyDeviceToHost); // 3d to host

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
    printf("%f,%f,%f\n",h_result_3d[40],h_result_3d[400],h_result_3d[4000]); 



    // freeing memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_result_3d);
    free(h_x);
    free(h_y);
    free(h_result);
    free(h_result_3d);
    free(h_x_cpu);
    free(h_y_cpu);
    free(h_result_cpu);
    


    return 0;
}