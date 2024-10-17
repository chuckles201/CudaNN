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

TODO
*** But how do we implement this on the GPU? ***