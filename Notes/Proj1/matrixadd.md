# Matrix addition Project

Lets write our first CUDA kernel to add two matrices with 2**20 elements each.

I'll be assuming some basic knowledge of C++, and explaining all the syntax that the CUDA API adds.

### 0.
Lets get some info about our GPU with this simple 'DeviceQuery'

* First I clone the NVIDIA cuda-samples github repo
* Then I run the device query file, and get some specs about my GPU
    <pre>
    ...
    // output of deviceQuery from official nvidia samples
    </pre>

    ### 1. We define a function that runs on a GPU like this:
    <pre>
    // Gpu Kernel definition
    __global__ myFunc()....

    // Running function <<<%n_blocks,threads_per_block>>>
    ...
    myFunc<<<%nblocks,nthreads>>>();
    ...
    </pre>

### 2. Now lets allocate some memory for the cuda
- First, we must use these CUDA functions for transfering memory from the host device (CPU) to the GPU, and then back, to write out our results

    <pre>
    int size = sizeof(float)*N;
    float *d_x, *h_x; // vector x on device and host
    ... // init h_x

    // Alloting memory, copying to memory
    cudaMalloc(&x, size); // allocate memory for d_x
    cudaMemcpy(d_a,h_a,size CudaMemCpyHostToDevice);

    // Copying back to CPU to read results
    cudaMemcpy(h_a,d_a,CudaMemCpyHostToDevice)


    // freeing memory
    cudaFree(x);
    cudaFree(y);
    </pre>

### 3. We have memory on the GPU, now we just need to execute our vector add function! 

- Step 1: defining threads
    - Threads are individual operations running in parallel.
    - A given threads index is defined by its threadIdx and its blockIdx (x,y), so we can acess its overall position with y*x + x. Think of each dimension as 'container' for other dimensions, so each +1 in this higher dimension is adding the entire lower dimension. So we can easily abstract this and index with 'strides' as defined above.
    - To be more specific, each thread in a thread block actually may have up to 3 chordinates denoting its position inside the threadblock, and each threadblock has three dimensions as well. This is conceptually the same, we just add a 2nd/3rd dimension to thread and block indexing to more easily work with convolutions and reason about memory structure.

- Now, coding this with CUDA:

1. Defining our dimensions
2. using cudas 'dim3' object type
3. executing function!



    <code>Arbitrary dim. size for indexing

        __global__ void info() {

            int block_id = 
                gridDim.x +
                blockIdx.y * gridDim.x + // multiplying offset
                blockIdx.z * gridDim.x * gridDim.y;

            int thread_id = threadIdx.x +
                threadIdx.y * blockDim.x +
                threadIdx.z * blockDim.y * blockDim.x;

            int index = blockDim.y * blockDim.x * blockDim.z *
                block_id + thread_id;   // block offset + thread id  

            printf("Index: %d | Thread ID: %d | (%d,%d,%d) | Block ID: %d (%d,%d,%d)\n",index, thread_id,threadIdx.x, threadIdx.y,threadIdx.z,block_id,blockIdx.x,blockIdx.y,blockIdx.z);

        }

        void vector_init() {

            ;
        }

        int main(){

            // defining block&thread dimensions (arbitrary #)
            const int b_x = 2, b_y = 3, b_z = 4;
            const int t_x = 4, t_y = 4, t_z = 4; //64

            int blocks_per_grid = b_x*b_y*b_z;
            int threads_per_block = t_x*t_y*t_z;

            // // allocating memory / transfering to device
            // float *x,*y,*z; // host variables
            // float *d_x,*d_y,*d_z; // device (GPU) variables

            // cudaMalloc(x,d_x,cudaMemcpyHostToDevice);
            // cudaMalloc(y,d_y,cudaMemcpyHostToDevice);
            // cudaMalloc(z,d_z,cudaMemcpyHostToDevice);

            // executing function
            dim3 BLOCK_DIM = {b_x,b_y,b_z};
            dim3 THREAD_DIM = {t_x,t_y,t_z};
            info<<<BLOCK_DIM,THREAD_DIM>>>(); // executes on GPU!
            cudaDeviceSynchronize();

            return 0;
        }
    </code>

This runs and we can verify our results. Note that
we are not actually changning any memory, we're just looking at how threads behave. Each thread will execute the function and the only thing different is that it has a different blockIdx and threadIdx for each dimension.


### 4. Actually coding the matrix addition

Ok, now lets use this knowledge to write our matrix-addition kernel,
we'll be using 1d threads for simplification purposes.

--> A given thread will have thread id x, and block id x
--> We'll be treating matrices as 1d arrays and indexing with the given shape if needed only at the end

Naked Matrix-addition kernel: TODO: fix

    <code>Code:

        __global__ void addGPU(int size,float* mat1, float* mat2,float*result){
            //ovr. index
            int index = threadIdx.x + blockIdx.x * blockDim.x;

            if (index < size) { // if inside bounds
                result[index] = mat1[index] + mat2[index];
            }
        }

        ...

        // allocating memory (x&y are matrices)
        float *h_x,*h_y, *h_result; // memory for matices
        float *d_x, *d_y, *d_result; // memory on GPU

        int dim1 = matX; // dimensions of matrices
        int dim2 = matY;
        int matsize = matX*matY; // size of unravelled matrix
        int size = matsize * sizeof(float); // memory

        // initialize matrices
        matrix_init(matsize,h_x,h_y,h_result); // changing memory locations

        // allocating,copying memory to/on GPU
        cudaMalloc(&d_x, size); // putting memory into GPU
        cudaMalloc(&d_y,size);
        cudaMalloc(&d_result,size);
        cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_result,h_result,size,cudaMemcpyHostToDevice);


        // executing on GPU
        /*
        we have 10k elements to multiply and 256 threads-per-block
        to ensure execution works, we must launch 10k/256 = 40 blocks (rounded up)
        */
        int blockSize = THREADS;
        int numBlocks = (matsize + blockSize -1) / blockSize;
        addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
        cudaDeviceSynchronize();

        ...
    </code>



### 5. TODO Benchmarking and error checking