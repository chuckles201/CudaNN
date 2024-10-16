# CUDA Background

- CUDA is designed to allow both small-scale, and large-scale utilization of parallelization, which has revolutionized certain areas of computing
- It allows developers complete control over indidvidual thread which are grouped into thread-blocks
- The beauty of CUDA, is that these thread blocks, can then be executed by any CUDA-GPU in any order, concurrently, or sequentially!

        -Analogy: Thread-blocks are like legos: they allow people to build programs with small building blocks and without much upfront cost and the ability to scale. A non-CUDA program is like a pre-built house that you cannot modify or scale in any way, but only works for a specific use-case.

Here is a group of thread-blocks executing on GPUs with varying lms (showcasing scaling-ability):
![thread-execution](automatic-scalability.png) 