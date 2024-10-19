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
        m[i] = 10.0;

    }
}

// SMEM sharing with DRAM coalescing 
__global__ void smemMatMul(int m, int n, int k, float *A, float *B, float *C, float alpha, float beta) {

    // defining where we are in block
    const int blockRow = blockIdx.x * BLOCK_SIZE;
    const int blockCol = blockIdx.y * BLOCK_SIZE;

    const int threadRow = threadIdx.x / BLOCK_SIZE;
    const int threadCol = threadIdx.x % BLOCK_SIZE;

    // Starting pointers off at their first 'slide'
    // make sure cont. threads load from same col
    //-for global mem (DRAM) coalescing
    A += blockRow * K;
    B += blockCol;
    C += blockRow*N + blockCol;

    // allocate smem
    __shared__ float As[SMEM_CHUNK_SIZE*BLOCK_SIZE];
    __shared__ float Bs[SMEM_CHUNK_SIZE*BLOCK_SIZE];

    // 'sliding block' loop
    float tmp = 0.0; // storing our value
    for (int blck = 0; blck < K; blck += SMEM_CHUNK_SIZE) {
        // loading memory DRAM-->SMEM
        As[threadIdx.x] = A[threadCol + threadRow * SMEM_CHUNK_SIZE];
        Bs[threadIdx.x] = A[threadCol + threadRow * BLOCK_SIZE];
        __syncthreads(); // making sure all memory loaded

        // getting pointer ready for next load
        A += SMEM_CHUNK_SIZE;
        B += SMEM_CHUNK_SIZE * N;

        for (int l = 0; l < SMEM_CHUNK_SIZE; l++) {
            tmp += As[threadRow*SMEM_CHUNK_SIZE + l] * 
                   Bs[threadCol + l*BLOCK_SIZE];
        }

        __syncthreads();// need all memory 

    }
    C[threadRow*N + threadCol] = tmp*alpha + C[threadRow*N + threadCol]*beta;
    
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
