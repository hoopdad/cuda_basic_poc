# cuda_basic_poc
A basic C++ example using the NVIDIA CUDA library for math, comparing CPU and GPU performance for the same tasks.

Setup on a system with an NVIDIA GPU. I am running WSL on Windows and installed CUDA with:

```bash
sudo apt-get install cuda
```

## Vector Add

Very simple arithmetic. 3 arrays are created. The first array is assigned a value; the second array is double that value. The third array is the addition of the two.

The CPU is more than capable of this and the overhead of moving data to GPU memory is relatively slow. So the CPU is actually faster in this case.

Increasing the array size might make the GPU perform better than CPU.

## Matrix math

The arithmetic here gets more complicated, and the GPU outperforms the CPU.