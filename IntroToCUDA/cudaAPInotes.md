# CUDA extra notes

- We should utilize CUDA's special device functions like cosf or fused mul-add, instead of cos. We should explicitly state in our kernels when we do this. More information in the CUDA math manual

### Getting information on execution

1. For each operation you wish to test:

        #include <nvtx3/nvToolsExt.h> // for debudding

        nvtxRangePush("Operation1");
        ...
        nvtxRangePop(); //operation tracked

Then we can simple run in the terminal:

        nvcc -o test matmul.cu -lnvToolsExt

        nsys profile -o ./test
        nsys-ui profile
        or
        ncu --set detailed -o profile ./test
        ncu-ui profile


        nvcc -arch=sm_89 -Xptxas -v my_program.cu -o run --> registers/smem info
        
this will give us simple stats about our program!

2. Useful commands for debugging memory usage/compiler code
        nvcc -ptx matmulV3.cu --> full compiler code

        nvcc -cubin example_cuda_kernel.cu -o example_cuda_kernel.cubin
        cuobjdump -sass example_cuda_kernel.cubin > example_cuda_kernel.sass


