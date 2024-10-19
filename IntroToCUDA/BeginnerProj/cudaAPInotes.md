# CUDA extra notes

- We should utilize CUDA's special device functions like cosf or fused mul-add, instead of cos. We should explicitly state in our kernels when we do this. More information in the CUDA math manual

### Getting information on execution

1. For each operation you wish to test:

        #include <nvtx3/nvToolsExt.h> // for debudding

        nvtxRangePush("Operation1");
        ...
        nvtxRangePop(); //operation tracked

2. Then we can simple run in the terminal:

        nvcc -o test matmul.cu -lnvToolsExt

        nsys profile --stats=true ./test

this will give us stats about our program!