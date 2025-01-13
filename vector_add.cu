#include <stdio.h>
#include <iostream>
#include <chrono>

__global__ void add(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void gpu_add(int arraySize, int a[], int b[], int c[]) {

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // Copy results from device to host
    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < arraySize; ++i) {
        // printf ("%d: %d + %d = %d \n", i, a[i], b[i], c[i]);
        if (c[i] != a[i] + b[i]) {
            printf("Error: c[%d] = %d, expected %d\n", i, c[i], a[i] + b[i]);
            return;
        }
    }

    printf("GPU Vector addition successful!\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void cpu_add(int arraySize, int a[], int b[], int c[]) {
    for (int i=0;i<arraySize; i++) {
        c[i] = a[i] + b[i];
    }
    // Verify results
    for (int i = 0; i < arraySize; ++i) {
        // printf ("%d: %d + %d = %d \n", i, a[i], b[i], c[i]);
        if (c[i] != a[i] + b[i]) {
            printf("Error: c[%d] = %d, expected %d\n", i, c[i], a[i] + b[i]);
            return;
        }
    }

    printf("CPU Vector addition successful!\n");
}

int main() {
    const int arraySize = 100000;
    int a[arraySize], b[arraySize], c[arraySize];

    // Initialize input arrays (e.g., with random values)
    for (int i = 0; i < arraySize; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_add(arraySize, a, b, c);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_add(arraySize, a, b, c);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Calculate the execution time
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    // Print the execution time
    std::cout << "GPU Execution time: " << gpu_duration.count() << " microseconds" << std::endl;
    std::cout << "CPU Execution time: " << cpu_duration.count() << " microseconds" << std::endl;


    return 0;
}
