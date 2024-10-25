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
#define M 4096
#define N 4096
#define K 4096

#define CEIL_DIV(m,n) ((m+n-1) / n)

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


// template defines how large the chunks we are loading from
// in our slides, and how many results we're calculating per thread

// in launch_bounds, we specify: {maxthread/block,minblcks per sm}
// threads/block will be maxed at the size/resultsperthread
template<const int BM,const int BN,const int TN,const int TM, const int BK>
__global__ void __launch_bounds__((BM*BN)/(TM*TN),1)
blockTile2D(int m, int n, int k, float *A, float *B, float *C, float alpha, float beta){

    // relative position of thread in block,
    // will be mutliplied by 8 to account
    // for multiple results per block
    const int threadRow = threadIdx.x / (BM/TM);
    const int threadCol = threadIdx.x % (BN/TN);

    // position that block is in
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // calculating results per threads
    const int numResults = BM*BN;
    const int resultsPerThread = TM*TN;
    const int numThreads =numResults/resultsPerThread;
    assert(blockDim.x == numThreads);

    // moving pointers to starting positions
    A += blockRow*BM*k;
    B += blockCol*BN;
    C += blockCol*BN+blockRow*BM*n;

    // shared memory for each BM,n*BK block
    __shared__ float As[BK*BM]; // transposed!
    __shared__ float Bs[BK*BN];

    // registers for reusing cols/rows
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // for TMxTN block of results
    float threadResult[TM*TN] = {0.0};

    // calculating where smem should be loaded
    // threads should span across collumns
    // so that gmem coalescing is possible
    const int innerColA = threadIdx.x % BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN;
    const int innerRowB = threadIdx.x / BN;

    // strides should be the number we need
    // to have loaded all cols
    const int strideA = numThreads / BK; // TRANSPOSED
    const int strideB = numThreads / BN;


    // outer block-slide loop
    // shoulder over-step and account 
    // for mismatch shapes
    for (int blckIdx = 0; blckIdx < k; blckIdx += BK) {

        // loading As and Bs with BM*BK/numthreads block slides
        // sliding down cols to preserve gmem coalescing
        // A is transposed
        for (int aOffset = 0; aOffset < BK; aOffset += strideA){
            // transpose our memory that we load
            As[innerRowA+aOffset+innerColA*BM]
             = A[(innerRowA+aOffset)*k+innerColA];
        }
        for (int bOffset = 0; bOffset < BK; bOffset += strideB){
            Bs[innerColB+(innerRowB+bOffset)*BN]
             = B[innerColB+(innerRowB+bOffset)*n];
        }
        __syncthreads(); // load memory before acessing for ops

        // moving pointers to next position for next slide
        A += BK;
        B += BK*n;

        // outer dot-product loop
        for (int dotIdx=0; dotIdx<BK; dotIdx++) {
            // saving registers and doing 
            // each ith dotIdx of B with all
            // jth dotIdx of A for maximum reuse!
            for(int mIdx = 0; mIdx < TM; mIdx++){
                regM[mIdx] = As[dotIdx*BM+threadRow+mIdx];
            }
            for(int nIdx = 0; nIdx < TN; nIdx++){
                regN[nIdx] = Bs[threadCol*TM + nIdx+dotIdx*BN];
            }

            // now for each B, doing first dot part
            // of all A's storing this, and re-adding
            // for next dotIdx.
            for (int nInd = 0; nInd < TN; nInd++){
                for (int mInd = 0; mInd < TM; mInd++){
                    threadResult[nInd + mInd*TN] +=
                    regM[mInd] * regN[nInd];
                }
            }
        }
        __syncthreads(); // before loading in next block
        

    }
    // storing results in C
    // already displaced by block, now
    // just need to do by threads pos. in block
    for(int mIdx = 0; mIdx < TM; mIdx++) {
        for (int nIdx = 0; nIdx < TN; nIdx++){
            C[nIdx+threadCol*TN+(threadRow*TM+mIdx)*n] =
            alpha*threadResult[mIdx*TN+nIdx] + 
            beta*C[nIdx+threadCol*TN+(threadRow*TM+mIdx)*n];
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
    
    //defining how we launch
    const int BM = 128;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8; // 16x16 block...
    const int BK = 8;

    dim3 blockSize = {BM*BN/(TM*TN)}; //1d to manip.
    // how many blocks we need
    dim3 numBlocks = {CEIL_DIV(M,BM),CEIL_DIV(N,BN)};


    // warmup
    for (int i = 0; i < 4; i++){
        blockTile2D<BM,BN,TN,TM,BK> // for template
        <<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
        cudaDeviceSynchronize();
    
    }

    // running 
    //nvtxRangePush("MatMul");
    double start = get_time();
    blockTile2D<BM,BN,TN,TM,BK>
    <<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
    cudaDeviceSynchronize();
    double end = get_time();
    //nvtxRangePop();


    // checking results
    printf("Time: %fms\n",(end-start)*1000.0);
    cudaMemcpy(c_h,c_d,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    printf("GPU Results: (%f,%f,%f,%f)\n",c_h[30],c_h[10000],c_h[1674445],c_h[M*N-1]);

    printf("%d.%d.%d\n",blockSize.x,numBlocks.x,numBlocks.y);
    

    // freeing memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);


    return 0;

    

}
