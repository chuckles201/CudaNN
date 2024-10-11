# GPU architecture (CUDA)

**Thread/execution hierarchy**
- GPU's have memory, and the ability to interact with the CPU, however the highest bandwith memory (and also the lowest storage) is in the GPUs SMs, which is where the magic happens.
- Threads (individual processes/operations) are grouped into 'thread-blocks'. Each SM runs 1 thread-block. To fully utilize a GPU, we need multiple thread-blocks for each SM (multiple sequences). We also need many 100s of threads per thread-block to fully utlize the SMs' recources.

        Example: A thread like a tab in your browser: it is different from the other tabs (threads), and not interefering with their tasks, while sharing the same memory from the host machine!
- It is also important to note that the more threads you launch, the smaller percentage that are in the 'tail-wave', where memory doesn't exactly fit. This means your getting more bang for your buck.
--------------------------------------------------------
## GPU preformance
***Limiting factors: memory bandwith, math bandwith, latency***

    Example: a GPU is executing a function that takes T_mem (time to read/write from memory) and T_math to execute. These functions are 'overlapped' meaning they are preformed simultaneously (one doesn't need to weight for the other). Therefore, the limiting factor in time is the longer one. This means a GPU can either be memory-bound or math-bound concerning its speed.

- Time on math processor: operations/math-bandwith, while memory speed is bytes-acessed/memory-bandwith.
 
    So operations/math-bandwith > bytes-acessed/memory-bandwith, is a math-limited system. We can rearange this to be: 
    (ops/bytes) > (BW_math / BW_mem)
    We can think of this as the proportion of math-byte operations relative to the processors speeds (bandwiths).

        Here are some examples of this:
        1. Linear layer (512,1024,4096) --> 321Flops/B (ops/bytes), so limited usually by arithmetic (set BW's)
        2. Linear layer (1,1024,4096) --> 1 FLop/B, so limited by memory!
        3. Relu: 0.25 flops/B : memory limited
        4. 'max pooling 3x3 window, unit stride' 2.25 flops/b: memory limited
        5. Layernorm : 10flops/b : memory limited

- Note that in these examples we are naively counting the 'ops', because we are only counting the arithmetic instructions, not the memory specifications/control instructions

- The final constraint is latency. Latency occurs when we underuse or 'undersaturate' the GPU, and we are not utilizing it to its full potential

        Example: We havea  V100 GPU which acesses 16 bytes and preforms 16k math operations (ops/bytes = 1k), and since the BW_math / BW_mem is less than 1k, we are math-limited. However, we only launch 1 'thread', meaning we only utilize part of the cores available to us: so our path potential is much, much lower. 
        
        Furthermore, if our algorithm reads elements multiple times we can acess them more effeciently, reducing mathematical intensity, and therefore ops/bs
----------------------------------------------------------
***Summary of constraints on GPU's***: 
1. We can assume that GPU's preform memory and mathematical operations at the same time, so the entire time taken to execute a function is: max(tmem,tmath)
2. We can easily calculate which operation will take longer given knowledge about the 'bandwith' of mathematical operations and reading/writing to memory: {(ops/bytes) >or< (bw_math/bw_mem) = (bw_math/ops) >or< (bw_mem/bytes)}
3. However, we must account for the fact that sometimes GPUs math ops are underutilized, and therefore, 
may be much slower than we first accounted for when considering the GPUs maximum bandwith. In this case the constraint is latency.
-------------------------------------------------------------------------------
***Common DNN operations preformed on GPU***
Here are a list of some common operations that are used in DL/NNs

1. Elementwise operations: nonlinearities, addition, multipllication, exponentiation, etc. Usually are memory-bound because each element only goes through a small operation
2. Dot-Product operations: for each element compute many additions/multiplications (dot prod of vectors); many more ops/byte. For example: convultions and Linear layers. Operations can be math-limited if large enough
3. Reduction operations: involve all values over a tensor: softmax, normalization, etc.

-------------------------------------------------------------------------------------
***Key takeaways***

(for approximating preformance limitations on single function...)
1. Know for your GPU the number of SMs, and how many threads they can hold (w/thread blocks). As per nvidia, you should have 4x as many thread blocks as sm, and a few hundred threads per thread-block
2. Know the bw_math/bw_memory ratio (or ops/bytes) to know how many ops/byte you should aim for when calculating arithmetic intensity
