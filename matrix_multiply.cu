#include <iostream>
#include <chrono>
#include <cstdlib> 

__global__ void matrixMultiply(float* result, const float* matrix1, const float* matrix2, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float sum = 0.0f;
        for (int k = 0; k < cols; ++k) {
            sum += matrix1[row * cols + k] * matrix2[k * cols + col];
        }
        result[row * cols + col] = sum;
    }
}

void gpu_matrix_multiply(int rows, int cols, float* matrix1, float* matrix2, float* result) {
    // Allocate device memory
    float* d_matrix1, *d_matrix2, *d_result;
    cudaMalloc((void**)&d_matrix1, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_matrix2, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_result, rows * cols * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_matrix1, matrix1, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size as needed
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_result, d_matrix1, d_matrix2, rows, cols);

    // Copy results from device to host
    cudaMemcpy(result, d_result, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);
}

void cpu_matrix_multiply(int rows, int cols, float* matrix1, float* matrix2, float* result) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i * cols + j] = 0.0f;
            for (int k = 0; k < cols; ++k) {
                result[i * cols + j] += matrix1[i * cols + k] * matrix2[k * cols + j];
            }
        }
    }
}

int main() {
    const int rows = 1024;
    const int cols = 1024;

    float* matrix1 = new float[rows * cols];
    float* matrix2 = new float[rows * cols];
    float* result_gpu = new float[rows * cols];
    float* result_cpu = new float[rows * cols];

    // Initialize matrices (e.g., with random values)
    for (int i = 0; i < rows * cols; ++i) {
        matrix1[i] = static_cast<float>(rand()) / RAND_MAX; 
        matrix2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_matrix_multiply(rows, cols, matrix1, matrix2, result_gpu);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matrix_multiply(rows, cols, matrix1, matrix2, result_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // Calculate the execution time
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);

    // Print the execution time
    std::cout << "GPU Execution time: " << gpu_duration.count() << " microseconds" << std::endl;
    std::cout << "CPU Execution time: " << cpu_duration.count() << " microseconds" << std::endl;

    // Verify results (optional)
    // ...

    delete[] matrix1;
    delete[] matrix2;
    delete[] result_gpu;
    delete[] result_cpu;

    return 0;
}