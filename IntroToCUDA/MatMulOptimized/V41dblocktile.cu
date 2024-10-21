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


// Defining dimensions of  m1, n2, m2/n1
#define M 4092
#define N 4092
#define K 4092

#define BLOCK_SIZE 32
#define SMEM_CHUNK_SIZE 32

// time function
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_nsec * 1e-9 + ts.tv_sec;

}

// matrix init function:
void matrixInit(float *m, int size) {
    for (int i = 0; i < size; i++) {
        m[i] = 2.0;

    }
}

// smem cacheing + global mem coalescing + 1d blocktiling
__global__ void smemMatMul(int m, int n, int k,float* A, float* B, float *C,float alpha,float beta) {
    // dimensions for warptile
    const int BN = 64;// how many cols in warptile
    const int BM = 64; // length of rows
    const int BK = 8; // length m1, width m2
    const int TM = 8; // num results per thread

    // block positions in grid
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    
    // thread positions in block 
    // col should access continous memory in matrix to group loads
    const int threadRow = blockIdx.x / BLOCK_SIZE;
    const int threadCol  =blockIdx.x % BLOCK_SIZE; 

    // advancing pointers to starting slide position
    A += cRow * BLOCK_SIZE;
    B += cCol * BLOCK_SIZE* N;
    C += cRow*BLOCK_SIZE + cCol*BLOCK_SIZE*N;

    // smem caches
    __shared__ float As[BM*BK];
    __shared__ float Bs[BN*BK];

    // now index for our given warptile inside block
    assert(BM*BK == blockDim.x); // block will work in this with threads
    assert(BN*BK == blockDim.x);

    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    // outer block loop
    float tmp[TM] = {0.0}; // storing dot prod current sum on register
    for (int i = 0; i<K; i+=BK) { 
        // load smem caches with indiv. threads
        // notice how we allign continuity with threadidx (threadcol)
        As[threadCol + M*threadRow] = A[threadCol+threadRow*m];
        Bs[threadCol + M*threadRow] = B[threadCol+threadRow*n];
        __syncthreads();

        // adv. blocktile slide
        A += BK;
        B += BK*N;

        // per-thread results
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // making dot product the outside loop, so we can reuse 
            for (int l = 0; l < )
        }

        
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

    float alpha = 1;
    float beta = 1-alpha;
    
    // dimensions for x/y
    dim3 blockSize = {BLOCK_SIZE*BLOCK_SIZE}; //1d to manip.
    dim3 numBlocks = {(M+BLOCK_SIZE-1)/BLOCK_SIZE,
                      (N+BLOCK_SIZE-1)/BLOCK_SIZE};

    

    // warmup
    for (int i = 0; i < 4; i++){
        smemMatMul<<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
        cudaDeviceSynchronize();
    
    }

    // running and debugging
    //nvtxRangePush("MatMul");
    double start = get_time();
    smemMatMul<<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
    cudaDeviceSynchronize();
    double end = get_time();
    //nvtxRangePop();


    // checking results
    printf("Time: %fms\n",(end-start)*1000.0);
    cudaMemcpy(c_h,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    printf("GPU Results: (%f,%f,%f,%f)\n",c_h[30],c_h[10000],c_h[1000000],c_h[99999]);
    

    // freeing memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);


    return 0;

    

}
