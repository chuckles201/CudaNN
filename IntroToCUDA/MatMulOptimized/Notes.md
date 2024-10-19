# Matmul

### Goal: Acheive close to state-of-the-art preformance with custom kernel that is understandable and simple

- I will be following an Anthropic Kernel engineer's guide (siboehm)


## Kernel v1: Naive

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

### Information about memory and FLOPS
- Lets keep track of how many operations we are doing, and how much memory we are reading. (assume we are multiplying 2 4092^2 matrices and then adding a 4092^2 matrix)
- Remember that since memory and math overlap in GPU's, the thing that takes longer will be our bottleneck(giving us ops/bytes ratio)

1. FLOPS for matmul: 2*4096^3 (for each element we have a fused multiply-add which multiplies and adds to a running sum in one go) + 4092^2 flops for addition
 = 137.5G flops
2. 67.1MB of memory to read 3*(4029^2), and 201MB 4092^2 write
3. Our GPU (A100) has 19.5TFLOPS of compute bandwith for F32, and 1,935GB/S memory bandwith. 

This means that our read/write to memory will take about 268/1,935 miliseconds (0.13) to read/write, and 137.5/19.5 (7) miliseconds


### ***Issue 1*** memory acess pattern of naive kernel
- Two kernels in the same block sometimes load the same row (ie: 0,0, 0,1), or the same collumn, however, they are doing this completely independently of eachother. This means we are reading much more memory than necessary.
- This means we acheive much lower FLOPS (time); but in reality, our memory is limiting us hugely.
    (our flops are : 2*4096^3+4096^2 / 0.43 =>300Gflops :( )
#### Warps
- Each SM processes a thread block, and 'warps' are the actual groups of threads that are processed together. They are 32 threads long, and are indexed on the 'unravelled matrix' (think indexing dimensions in memory: x*dimy + y). So in our 32x32 blocks, there are 32 warps.
- However, while we were discussing memory in terms of 'row-major' before, where y represented continous memory (or the last dim), now we are collumn-major, where x represents continous memory  (actuall_id = x + y*dimx)

This matters, because in a warp, multiple sequential memory acesses ***can be combined into 1***

## Kernel V2 Coalescing warp memory
- So, we clearly need similar warps to acess similar rows, meaning that our dimensions in our thread blocks should match those of our matrices.
- In this code, each threadIdx actually controls the y-axis on the grid, but this is what controls the continuous memory in the matrix (its inverted), so we allign the warp memory and the loaded memory from the rows that we can share within a warp


        // Coalescing global memory
        __global__ void sgemm_coalesced(int m,int n,int k,float alpha, float *A, float* B,float beta,float *C){

            // x down and to the side
            const int x = blockIdx.x*BLOCK_SIZE + 
            (threadIdx.x/BLOCK_SIZE); // how far down within a block (to side!)
            // side and down (reverse on grid)
            const int y = blockIdx.y*BLOCK_SIZE + 
            (threadIdx.x%BLOCK_SIZE); // how far to l/r in block (down!);
            

            if (y < n && x < m) {
                // dot product of ith row m1/ jth col i2
                float tmp = 0;
                for(int l = 0; l < k; l++) {
                    tmp += A[x*n + l] * B[y + l*n];
                }
                C[x*n + y] = tmp*alpha + beta*C[x*n+y]; // adding and multiplying
            }

        }
- We acheive about 3.5x more flops here!
> [NOTE]
>
> Here, we only coalesce the global memory (meaning loading from our ram to our sms). Dram holds the most memory, and the subsequent lower levels of memory are smaller, but faster.


![GPU architecture](memory-hierarchy-in-gpus.png)

## Kernel V3

- Looking at this picture, we can see that smem is the local memory that each SM stores, meaning thread blocks can acess it.
- Intuitively, this on-chip memory is much faster, with lower latency, due to the the fact that it is physically closer to the threads which are accessing from it.
- So, ideally we would load the entire matrix into the shared memory, but since the space is finite, we will load chunks and then do work on them with our threads in our block, and then load more. The fast speed of the smem more than makes up for having 2-loading periods (for threads, and for the smem)

1. My SM supports about 164KB per load.
2. So, given a 32x32 thread-block, we're going to sequentially load the jth collumn of m2, and the ith collumn of m1. Since we need to load the entire 'K' row for m1 and col for m2, but only the 32 long/wide chunk (think of the 'strip' that we target with our thread-block), we basically scan across m1 loading 32X32 chunks of the row-strip, and the same for m2 with the collumn strip. 

Cacheing memory with smem (each thread loads a number)

        __global__ void smem_shared(int m,int n,int k,float alpha, float *A, float* B,float beta,float *C){
            // we load in chunks to our shared memory for a block to work on
            // for our A matrix, 32 high, 64 wide
            // for B, 32 wide, 64 long
            const int cRow = blockIdx.x; // pos. in grid
            const int cCol = blockIdx.y;

            const int threadRow = threadIdx.x / BLOCK_SIZE; // what y would be
            const int threadCol = threadIdx.x % BLOCK_SIZE;

            // adv. pointers to 'starting positions'
            A += cRow * BLOCK_SIZE * K;
            B += cCol * BLOCK_SIZE;
            C += cRow*BLOCK_SIZE*N + cCol*BLOCK_SIZE;

            float tmp = 0.0; // store temporary dot prod.

            // memory for local cache --> not a ptr-
            __shared__ float As[BLOCK_SIZE*SMEM_CHUNK_SIZE];
            __shared__ float Bs[BLOCK_SIZE*SMEM_CHUNK_SIZE];

            // outer-loop to advance our smem-block
            for (int bkIdx = 0; bkIdx < K; bkIdx+=SMEM_CHUNK_SIZE) {
                // each thread loads a part of As and Bs
                // as threadIdx increments, we want to increment-
                //-global memory for global coalescing to work
                As[threadRow*BLOCK_SIZE + threadCol] = A[threadRow*K + threadCol]; // changes by 1 with collumn
                Bs[threadRow*BLOCK_SIZE + threadCol] = B[threadRow*N + threadCol]; // skipping over all other elements in A/B

                __syncthreads(); // finish memory loading

                // updating A/B positions to start of next block
                A += SMEM_CHUNK_SIZE;
                B += SMEM_CHUNK_SIZE*N;

                // dot product for  (temporarily stored in smem) block
                for (int l = 0; l<SMEM_CHUNK_SIZE; l++) {
                    tmp += As[threadRow*SMEM_CHUNK_SIZE + l] * 
                    Bs[threadCol + l*SMEM_CHUNK_SIZE];
                }
                __syncthreads();

            }
            C[threadRow*N + threadCol] = alpha*tmp + beta*C[threadRow*N + threadCol];

        }


 We acheive a ~35% spedup! (49ms --> 30ms)

> Note
>
> \_\_shared\_\_ means smem being loaded in a pointer
> __synchthreads() is used so all the threads finish their jobs before the next step because each thread loads a part of the memory




### Further notes on memory usage
- Notice that although our GPU could handle much more than 32x32x4 byte chunks, we kept the number much lower. This is because, when we max out the SMEM on our SM, we loose the ability to store multiple blocks.

- Each SM is only executing one warp at a time, however having multiple blocks is very helpful because it switches between warps rapidly, and warps will often be waiting for memory to load or until other operations finish, so it is helpful to have high *occupancy*, or many thread blocks in queued in any given SM
    - However, there are limits to how many blocks we can load onto an SM at one time:
    1. As we saw SMEM is limited per SM, so we can't use too much per block
    2. Register Count
    3. Warp Count

> Registers are local variables that are local to the thread, and therefore even faster. then smem (shared within block) They store constants and information about what a thread should do. 

So, if we have too many registers, we can't run many threads concurrently, and if we use too much SMEM per block, we will exceed the maximum capacity. Also, there is a max number of warps we can have due to hardware limitations.


