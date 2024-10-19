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
__global__ void sgemm_coalesced(int m,int n,int k,float alpha, float *A, float* B,float beta,float *C){

    // x down, controls down
    const int x = blockIdx.x*BLOCK_SIZE + 
    (threadIdx.x/BLOCK_SIZE); // how far down within a block (to side!)
    // side, controls side
    const int y = blockIdx.y*BLOCK_SIZE + 
    (threadIdx.x%BLOCK_SIZE); // how far to l/r in block (down!);
    

    if (y < n && x < m) {
        // dot product of ith row m1/ jth col i2
        float tmp = 0;
        for(int l = 0; l < k; ++l) {
            tmp += A[x*k + l] * B[y + l*n]; // ith by x, jth by y
        }
        C[x*n + y] = tmp*alpha + beta*C[x*n+y]; // adding and multiplying
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
    matrixInit(c_h,M*N);

    // feeding information to device
    cudaMalloc(&a_d,sizeof(float)*M*K);
    cudaMalloc(&b_d,sizeof(float)*N*K);
    cudaMalloc(&c_d,sizeof(float)*M*N);
    cudaMemcpy(a_d,a_h,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b_h,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    cudaMemcpy(c_d,c_h,sizeof(float)*N*K,cudaMemcpyHostToDevice);
    //------------------------------------------------

    float alpha = 0.001;
    float beta = 1-alpha;
    
    // dimensions for x/y
    dim3 blockSize = {BLOCK_SIZE*BLOCK_SIZE}; //1d to manip.
    dim3 numBlocks = {(M+BLOCK_SIZE-1)/BLOCK_SIZE,
                      (N+BLOCK_SIZE-1)/BLOCK_SIZE};



    // running and debugging
    //nvtxRangePush("Operation1");
    double start = get_time();
    sgemm_coalesced<<<numBlocks,blockSize>>>(M,N,K,alpha,a_d,b_d,beta,c_d);
    cudaDeviceSynchronize();
    double end = get_time();
    //nvtxRangePop();




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
