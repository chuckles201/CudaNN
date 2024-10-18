// Including Kernel definitions
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvtx3/nvToolsExt.h> // for debudding

// other libs
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>


// Defining dimensions of  m1, n2, m2/n1
#define M 4092
#define N 4092
#define K 4092

#define BLOCK_SIZE 32

// time function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_nsec * 1e-9 + ts.tv_sec;

}

// matrix init function:
void matrixInit(float *m, int size) {
    for (int i = 0; i < size; i++) {
        m[i] = 10.0;

    }
}

// Naive kernel
__global__ void sgemm_naive(int m,int n,int k,float alpha, float *A, float* B,float beta,float *C){

    // position in mat for thread
    int x = threadIdx.x + blockIdx.x * blockDim.x; // row
    int y = threadIdx.y + blockIdx.y * blockDim.y; // col
    

    if (y < n && x < m) {
        // dot product of ith row m1/ jth col i2
        float tmp = 0;
        for(int l = 0; l < k; l++) {
            tmp += A[x*n + l] * B[y + l*n];
        }
        C[x*n + y] = tmp;
    }

}


int main() {
    //------------------------------------------------
    // memory on host
    float *a_h,*b_h,*c_h, *a_d,*b_d,*c_d;
    a_h = (float*)malloc(sizeof(float)*M*K);
    b_h = (float*)malloc(sizeof(float)*N*K);
    c_h = (float*)malloc(sizeof(float)*M*N);

    matrixInit(a_h,M*K);
    matrixInit(b_h,N*K);

    // feeding information to device
    cudaMalloc(&a_d,sizeof(float)*M*K);
    cudaMalloc(&b_d,sizeof(float)*N*K);
    cudaMalloc(&c_d,sizeof(float)*M*N);
    cudaMemcpy(a_d,a_h,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    //------------------------------------------------

    float alpha = 0.001;
    float beta = 0.001;
    
    dim3 blockSize = {BLOCK_SIZE,BLOCK_SIZE,1};
    dim3 numBlocks = {(M+BLOCK_SIZE-1)/BLOCK_SIZE,
                      (N+BLOCK_SIZE-1)/BLOCK_SIZE, 1};



    // running and debugging
    nvtxRangePush("Operation1");
    double start = get_time();
    sgemm_naive<<<numBlocks,blockSize>>>(M,N,K,alpha,a_d,b_d,beta,c_d);
    cudaDeviceSynchronize();
    double end = get_time();
    nvtxRangePop();




    // checking results
    printf("Time: %fms\n",(end-start)*1000.0);
    cudaMemcpy(c_h,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    printf("GPU Results: (%f,%f)\n",c_h[30],c_h[10000]);
    


    // freeing memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);


    return 0;

    

}
