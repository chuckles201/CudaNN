// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>

// other libs
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// global descriptors M1,N2, M2/N1
#define M 100
#define N 100
#define K 100

#define BLOCK_SIZE 16

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
            
        }


    }
}
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// GPU matmul!
__global__ void matMulGPU(float *A, float *B, float *C, int k,int m,int n) {
    // entire element (dot prod) for each thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {

        float dot = 0.0f;
        for (int i = 0; i < k; i++) {
            dot += A[row*k + i] * B[col + i*k];
        }
        C[row*n+col] = dot;
    }
}

/////////////////////////////////////////////////////////////////////

int main() {

    /////////////////////////////////////////////////////////////////////
    // simple CPU implementation 
    float *mat1_cpu, *mat2_cpu, *result_cpu;
    int matsize1 = M * K; // use for all
    int matsize2 = N * K;
    int matsize3 = M * N;
    mat1_cpu = (float*)malloc(sizeof(float)*matsize1);
    mat2_cpu = (float*)malloc(sizeof(float)*matsize2);
    result_cpu = (float*)malloc(sizeof(float)*matsize3);
    matrixInit(matsize1,mat1_cpu); // assinging values to memory positions
    matrixInit(matsize2,mat2_cpu);
    //////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////
    // GPU implementation
    // initializing memory
    float *result_h;
    float *mat1_d, *mat2_d, *result_d;
    result_h = (float*)malloc(sizeof(float)*matsize3);
    matrixInit(matsize1,mat1_cpu);
    matrixInit(matsize2,mat2_cpu);

    cudaMalloc(&mat1_d,sizeof(float)*matsize1);
    cudaMalloc(&mat2_d,sizeof(float)*matsize2);
    cudaMalloc(&result_d,sizeof(float)*matsize3);
    cudaMemcpy(mat1_d,mat1_cpu,sizeof(float)*matsize1,cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d,mat2_cpu,sizeof(float)*matsize2,cudaMemcpyHostToDevice);

    // grid and block dimensions
    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE-1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE-1) / BLOCK_SIZE
    );
    /////////////////////////////////////////////////////////////////

    // warmup steps
    for (int i=0; i<3; i++) {
        matMulCPU(mat1_cpu,mat2_cpu,result_cpu,K,M,N);
        matMulGPU<<<gridDim,blockDim>>>(mat1_d,mat2_d,result_d,K,M,N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(result_h,result_d,sizeof(float)*matsize3,cudaMemcpyDeviceToHost);
    printf("GPURESULTS: %f,%f,%f,%f\n",result_h[9999],result_h[250],result_h[300],result_h[0]);

    printf("CPURESULTS: %f,%f,%f\n",result_cpu[9999],result_cpu[0],result_cpu[5267]);

    // freeing memory
    free(mat1_cpu);
    free(mat2_cpu);
    free(result_cpu);
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(result_d);

}