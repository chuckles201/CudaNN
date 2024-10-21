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

// in this case, our blocks are 64 * 8,
// so two slides across collumn with warptiles
// and 8 slides down across rows
#define BLOCK_SIZE 64*8

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

// smem cache + coalescing loads + warptiles
__global__ void blockTile1D(int m, int n, int k, float *A, float *B, float* C, int alpha, int beta){
    // defining our blocktile sizes
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    assert(TM*BK*BM == BM*BN); // 8 results per thread

    // thread rows for a/b different shapes because diff. size
    // making sure cont. memory acessed by cont threadIdxs
    // this is the memory we are loading from each blocktile
    const int threadRowA = threadIdx.x / BK;
    const int threadColA = threadIdx.x % BK;
    const int threadRowB = threadIdx.x / BN;
    const int threadColB = threadIdx.x % BN;

    // starting off pointers at given block in grid
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BN * N + blockIdx.x*BN;

    // memory for smem cache
    __shared__ float As[BK*BM];
    __shared__ float Bs[BK*BM];

    // memory for each result
    float tmp[8] = {0.0};

    // block tile loop
    for (int blckIdx = 0; blckIdx < K; blckIdx += BK) {
        // loading memory into A and B dram-->smem
        // notice that both a&b can be coalesced
        As[threadRowA * BK + threadColA] = A[threadColA + threadRowA*K];
        Bs[threadRowB * BN + threadColB] = B[threadColB + threadRowB*N];
        __syncthreads(); // waiting until memory loaded to start ops

        // advancing pointers
        A += BK;
        B += BK*N;

        // dot product loop outside (given idx)
        // inside we switch rows of A, so we can re-use Btemp
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float Btemp = Bs[dotIdx*BN+threadColB]; // stored on register
            for (int rowIdx = 0; rowIdx < TM; rowIdx++) {
                tmp[rowIdx] += Btemp * As[rowIdx*BK + dotIdx];
            }
        }
        __syncthreads(); // wait until ops done to start loading mem again

    }
    // loading results of C
    // note how we skip 8 spaces below for the threads to fill in
    // with their 8 calculations
    const int cCol = threadIdx.x % BN;
    const int cRow = (threadIdx.x / BM)*8;
    for (int cIdx = 0; cIdx < TM; cIdx++) {
        // add below row for below calculations (1 warp has 8x1 result block)
        C[cCol + cRow*N + cIdx*N] = alpha*tmp[cIdx] + beta*C[cCol + cRow*N + cIdx*N];
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
    const int B1 = 8;
    const int B2 = 64;
    dim3 blockSize = {B1*B2}; //1d to manip.
    // since in down direction each thread does 8x more
    dim3 numBlocks = {(M+(B1*8)-1)/(B1*8),
                      (N+B2-1)/B2};

    

    // warmup
    for (int i = 0; i < 4; i++){
        blockTile1D<<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
        cudaDeviceSynchronize();
    
    }

    // running and debugging
    //nvtxRangePush("MatMul");
    double start = get_time();
    blockTile1D<<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
    cudaDeviceSynchronize();
    double end = get_time();
    //nvtxRangePop();


    // checking results
    printf("Time: %fms\n",(end-start)*1000.0);
    cudaMemcpy(c_h,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    printf("GPU Results: (%f,%f,%f,%f)\n",c_h[30],c_h[10000],c_h[1000000],c_h[16744463]);
    

    // freeing memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);


    return 0;

    

}
