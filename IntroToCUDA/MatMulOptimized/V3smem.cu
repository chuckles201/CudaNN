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

// smem cacheing + global mem coalescing
__global__ void smemMatMul(int m, int n, int k,float* A, float* B, float *C,float alpha,float beta) {

    // thread positions in block
    const int bCol = blockIdx.x;
    const int bRow = blockIdx.y;
    const int threadRow = threadIdx.x / BLOCK_SIZE;
    const int threadCol = threadIdx.x % BLOCK_SIZE; // should access same row

    // setting pointers to starting 'slide'
    A += bRow*BLOCK_SIZE*K;
    B += bCol*BLOCK_SIZE;
    C += bCol*BLOCK_SIZE + bRow*BLOCK_SIZE*N;

    // shared mem
    __shared__ float As[BLOCK_SIZE*SMEM_CHUNK_SIZE];
    __shared__ float Bs[BLOCK_SIZE*SMEM_CHUNK_SIZE];

    // loading data for each 'slide' and working on it
    float tmp = 0.0;
    for (int blck = 0; blck < K; blck += SMEM_CHUNK_SIZE) {
        // each thread loads its memory
        // make sure continous threads load continous mem. A/B
        As[threadCol + threadRow*SMEM_CHUNK_SIZE] = A[threadCol + threadRow*K];
        Bs[threadCol + threadRow*BLOCK_SIZE] = B[threadCol + threadRow*N];
        __syncthreads(); // wait for memory before operating!

        // update slides for next run
        A += BLOCK_SIZE;
        B += BLOCK_SIZE*N;

        // dot product with slide
        for(int l = 0; l < SMEM_CHUNK_SIZE; l++) {
            tmp += As[threadRow * SMEM_CHUNK_SIZE+ l] *
                   Bs[threadCol + BLOCK_SIZE*l];
        }
        __syncthreads(); // can't start loading new memory untill all done!


    }
    C[threadRow*N + threadCol] = alpha*tmp + beta*C[threadRow*N + threadCol];


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
