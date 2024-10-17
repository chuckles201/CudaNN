# Naive Matmul

Here we will execute a naive matrix multiplication. This is our first iteration, and is the most basic and easiest to understand. It  will serve as a benchmark for future kernels to see how fast we can speed up this operation.


### Components of matmul:
1. This operation is slightly more complex than an elementwise operation like addition and will require different elements.

Here is a CPU matmul with a for-loop:

    // CPU matmul
    void matMulCPU(float *A, float *B, float *C, int mn, int m1, int n2) {
        for (int i = 0; i<m1; i++){ // each collumn of new matrix
            for (int j =0; j<n2; j++) { // each row in given collumn
                float dot = 0; // dot product at given point
                for (int k = 0; k < mn; k++) {
                    dot += A[i*mn + k] * B[k*n2+i];
                }
                C[i*m1 + j] = dot;
                printf("ID:%d,(%d,%d) : %f\n", i*n2 + j,i,j,dot);
            }


        }
    }

This really just boils down too dotting rows with collumns and working with indexing. 

*** But how do we implement this on the GPU? ***
- We will have 2D indexing to make the multiplication more intuitive: basically each thread block has its own 'square' of limited size and we need multiple of these squares to get the entire matrix.

- Each thread-block will take a 32 by 32 section of the matrix. So, in order to be able to hit all parts of the resulting matrix, we get
M1 / block_size1, N2/ block_size2 blocks in the grid.

- Now, with our knowledge from vector addition, the code really doesn't change much, besides each thread having to do a dot product. We don't share memory between threads in thread blocks, worry about warps or optimization either (we'll be doing this extensively later)


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

