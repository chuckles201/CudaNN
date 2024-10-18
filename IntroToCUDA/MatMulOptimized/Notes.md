# Matmul

### Goal: Acheive close to state-of-the-art preformance with custom kernel that is understandable and simple

- I will be following Anthropic's lead Kernel engineer's guide


### Kernel v1: Naive

- We implement a naive matrix multiplication with a for-loop and 2-d threads and blocks. Each thread combines it's block ID and thread ID to get its overall point in the matrix (IE: thread.x + block.x*dim.x is its x chord), and then each element is computed as the dot product of the correspinding collumn of the 2nd matrix and row of the 1st. 


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

# TODO: add addition part

### Information about memory and FLOPS
- Lets keep track of how many operations we are doing, and how much memory we are reading. (assume we are multiplying 2 4092^2 matrices and then adding a 4092^2 matrix)
- Remeber that since memory and math overlap in GPU's, the thing that takes longer will be our bottleneck(giving us ops/bytes ratio)

1. FLOPS for matmul: 2*4096^3 (for each element we have a fused multiply-add which multiplies and adds to a running sum in one go) + 4092^2 flops for addition