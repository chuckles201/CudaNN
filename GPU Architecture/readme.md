# GPU Architecture

## What I'll provide
- Full coverage of theoretical aspects and limits of parallel computing including a breif history of GPUs.
- Description of individual elements of the GPU in the context of parrallelilzation
1. Going into more detail about how memory works in a GPU
2. Physical descriptions of hardware/GPU components
3. Describe how work is scheduled on a GPU, and go through some example assembly code to learn what is really happening when we run a kernel.

## Goal
- This part of the repo is meant to supplement the kernel optimizations that we will be doing, by giving us a deep understanding of the components of a GPU and an intuition for how to optimize a given kernel for a given problem.
- This ***wont*** be easy: I will try into depth on parts of the GPU not usually covered, and challenging any assumptions not based on understanding.