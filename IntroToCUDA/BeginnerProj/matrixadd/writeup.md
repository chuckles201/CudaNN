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

Naked Matrix-addition kernel: 

    // Including Kernel definitions
    #include <cuda_runtime.h>
    #include <cuda.h>

    // other libs
    #include <stdio.h>
    #include <iostream>
    #include <time.h>
    #include <stdlib.h>
    #include <iostream>

    // global vars
    #define matX 100
    #define matY 1000
    #define MATOVR matX * matY
    #define THREADS 256 // threads per-block
    #define m1 3 // just for checking for now
    #define m2 5

    // function for timing --> ignore error for clock_monotonic
    double get_time() {
        // creating holder for time
        struct timespec ts; 
        clock_gettime(CLOCK_MONOTONIC,&ts); // store at ts
        return ts.tv_sec + ts.tv_nsec * 1e-9; // nanoseconds and seconds


    }

    // defining Kernel
    __global__ void addGPU(int size,float* mat1, float* mat2,float*result){
        //ovr. index
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < size) { // if inside bounds
            result[index] = mat1[index] + mat2[index];
        }
    }

    // defining CPU function
    void addCPU(int matsize,float* mat1, float* mat2,float*result){

        for(int i = 0; i < matsize; i++) {
            result[i] = mat1[i] + mat2[i];
        }
    }

    // initializing memory for 2 matrices (on host)
    void matrix_init(int size, float *mat1,float *mat2,float*mat3) { // pointer arrays (increment memory location)
        for (int i = 0; i<size; i++){ 
            mat1[i] = (float)m1; // ith memory location in array
            mat2[i] = (float)m2;
            mat3[i] = (float)0;
        }
    }

    int main(){
        

        // allocating memory (x&y are matrices)
        float *h_x,*h_y, *h_result; // memory for matices
        float *d_x, *d_y, *d_result; // memory on GPU

        int matsize = matX*matY; // size of unravelled matrix
        int size = matsize * sizeof(float); // memory

        // initialize matrices
        h_x = (float*)malloc(size);
        h_y = (float*)malloc(size);
        h_result = (float*)malloc(size);
        matrix_init(matsize,h_x,h_y,h_result); // changing memory locations

        // allocating,copying memory to/on GPU
        cudaMalloc(&d_x, size); // putting memory into GPU
        cudaMalloc(&d_y,size);
        cudaMalloc(&d_result,size);
        cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice); // copying host to device
        cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice);
        cudaMemcpy(d_result,h_result,size,cudaMemcpyHostToDevice);


        // for executing on GPU ***
        int blockSize = THREADS;
        int numBlocks = (matsize + blockSize -1) / blockSize;

        // allocating memory with ptrs to array
        float *h_x_cpu, *h_y_cpu, *h_result_cpu;
        h_x_cpu = (float*)malloc(size);
        h_y_cpu = (float*)malloc(size);
        h_result_cpu = (float*)malloc(size);

        // intializing CPU
        matrix_init(matsize,h_x_cpu,h_y_cpu,h_result_cpu); // initializing matrices


        // 'warmup runs'
        for (int i=0; i<3; i++){
            addCPU(matsize,h_x_cpu,h_y_cpu,h_result_cpu);
            addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
            cudaDeviceSynchronize();
        }

        // CPU, GPU1D, GPU3D benchmark implementations
        // CPU run
        double time = 0;
        for (int i = 0; i<10; i++){
            double start = get_time();
            addCPU(matsize,h_x_cpu,h_y_cpu,h_result_cpu);
            double end = get_time();
            time += (end-start);

        }
        time = time/10;
        double time1 = time;

        printf("CPU AVG TIME: |%f miliseconds|\n",(float)time*1000);
        time = 0.0;

        // GPU run
        time = 0;
        for (int i = 0; i<10; i++){
            double start = get_time();
            addGPU<<<numBlocks,blockSize>>>(matsize,d_x,d_y,d_result);
            double end = get_time();
            time += (end-start);

        }
        time = time/10;
        double time2 = time;

        printf("GPU AVG TIME: |%f miliseconds|\n\n",time*1000.0);



        // Testing GPU accuracy
        cudaMemcpy(h_result,d_result,size,cudaMemcpyDeviceToHost); // copying info back to host
        bool correct = true;

        for (int i=0; i<matsize; i++) {
            if (h_result[i] != h_result_cpu[i]) {
                correct = false;
                break;
            }
        }
        printf("All values accurate? --> |%s|\n\n",correct ? "correct":"incorrect");

        printf("GPU Speedup vs. CPU : |%fx|",time1/time2);



        // freeing memory
        cudaFree(d_x);
        cudaFree(d_y);
        free(h_x);
        free(h_y);
        free(h_result);
        free(h_x_cpu);
        free(h_y_cpu);
        free(h_result_cpu);
        


        return 0;
    }

Ok! We:
1. Allocated memory in host device and copied to the GPU
2. Reasoned through the thread-structure (simple 1d in this case)
3. Time our results w/ ctime library

With this simple naive matrix addition, we acheived a 40x speedup (which would clearly scale for larger matrices)!

### 5. 3D vector addition function

Now, lets extend this this to addition of a 3d 'vector', and test this against our 1D vector addition.

Note:
* We will not only be intializing a 3D vector, but also 3D blocks. This essentially means that instead of doing operations of 'unrolled' vectors, we are using this dimension abstraction to preform a potentially more intuitive calculation (not in this example)

Here's the code:

    In progres... not working for some reason:((

