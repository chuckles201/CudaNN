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

#define CEIL_DIV(M,N) ((M+N-1) / N)

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
blockTile2D(int m, int n, int k, float *A, float *B, float *C, int alpha, int beta){
    // defining where we are in the matrix with
    // respect to blocks location
    const int bCol = blockIdx.x;
    const int bRow = blockIdx.y;

    // offsetting our pointers to col loc.
    A += bRow * BM * K;
    B += bCol * BN;
    C += bRow * BM * n + bCol*BN;

    // results/resultsperthread=total
    const int totalThreads = BM*BN / (TM*TN);

    // defining where thread is within a block'
    // 8x8 threads, with 8x8 area for each thread
    const int threadRow = (threadIdx.x / BK)*TM;
    const int threadCol = (threadIdx.x % BK)*TN;

    // defining smem for both A and B within 
    // each block
    __shared__ float As[BM*BK];
    __shared__ float Bs[BN*BK];

    // defining what positions each thread will
    // load memory into smem, and the strides.
    // threads span across cols for gmem coalescing!
    const int threadColA = threadIdx.x % BK;
    const int threadRowA = threadIdx.x / BK;

    const int threadRowB = threadIdx.x / BN;
    const int threadColB = threadIdx.x % BN;

    const int strideB = totalThreads / BN; // col-span
    const int strideA = totalThreads / BK;

    // defining our registers: we save a row of B
    // and a collumn of A for each dot prod for loop
    float regM[TM] = {0.0}; // store dot jth of A
    float regN[TN] = {0.0}; // store dot ith of B

    // register for our 2d block of results
    float threadResult[TM*TN] = {0.0};

    // outer block loop
    for(int blckIdx = 0; blckIdx < k; blckIdx += BK) {
        // loading As into smem cache
        // advancing down rows (strideA)
        for (int offsetA = 0; offsetA < BM; offsetA += strideA) {
            As[(offsetA+threadRowA)*BK + threadColA] = 
            A[(offsetA+threadRowA)*k + threadColA];
        }

        for (int offsetB = 0; offsetB < BK; offsetB += strideB) {
            Bs[(offsetB+threadRowB)*BN+threadColB] = 
            B[(offsetB+threadRowB)*n+threadColB];
        }
        __syncthreads(); // wait until memory loaded to access

        // advancing pointers
        A += BK;
        B += BK*n;

        // dot product loop on outside, so we can save
        // dotIdx ith from B, jth from A into registers
        // and add this to our total BK times
        for(int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // saving vals into registers
            for (int i = 0; i < TM; i++){
                // index at blocks  position to relevant
                // thread which computes TMxTN square
                regM[i] = As[dotIdx + (threadRow+i)*BK];
            }
            // load the collumn we're computing
            for (int i = 0; i < TN; i++){
                // index at blocks  position to relevant
                // A row. Same for all threads in a block
                regN[i] = Bs[i + dotIdx*BN + threadCol];
            }

            // computing by reusing each ith B to hit all
            // jth A's, TMxTN 1/dotlength parts of tile!
            // loading part of collumn and hitting all rows first
            for (int nIdx=0; nIdx < TN; nIdx++){
                for (int mIdx = 0; mIdx < TM; mIdx++){
                    // adding to appropriate spot
                    // dotIdxth part of the dot
                    threadResult[nIdx+mIdx*TN] +=
                    regN[nIdx] * regM[mIdx];
                }
            }
        }
        __syncthreads(); // now need to wait until all ops done

    } // end of block-slide

    // putting in results of C
    for(int mIdx = 0; mIdx < TM; mIdx++){
        for(int nIdx = 0; nIdx < TN; nIdx++){
            // sliding over block of C
            C[(threadRow+mIdx)*n+nIdx+threadCol] =
            alpha*threadResult[mIdx*TN+nIdx] + beta*C[(threadRow+mIdx)*n+nIdx+threadCol];
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
    const int B1 = 8;
    const int B2 = 8;
    dim3 blockSize = {B1*B2}; //1d to manip.
    // since in down direction each thread does 8x more
    dim3 numBlocks = {(M+63)/64,(N+63)/64};

    

    // warmup
    for (int i = 0; i < 4; i++){
        blockTile2D<64,64,8,8,8> // for template
        <<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
        cudaDeviceSynchronize();
    
    }

    // running and debugging
    //nvtxRangePush("MatMul");
    double start = get_time();
    blockTile2D<64,64,8,8,8>
    <<<numBlocks,blockSize>>>(M,N,K,a_d,b_d,c_d,alpha,beta);
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
