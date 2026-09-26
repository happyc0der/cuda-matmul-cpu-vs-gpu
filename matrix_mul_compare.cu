#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        float sum = 0.0f;
        for (int k = 0; k < width; k++)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// CPU function for matrix multiplication
void cpuMatrixMultiplication(float *A, float *B, float *C, int width)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
            {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

// GPU function for matrix multiplication
void gpuMatrixMultiplication(float *A, float *B, float *C, int width)
{
    int size = width * width * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (width + blockDim.y - 1) / blockDim.y);

    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    int width = 1024;
    int size = width * width;

    float *A = new float[size];
    float *B = new float[size];
    float *C_cpu = new float[size];
    float *C_gpu = new float[size];

    for (int i = 0; i < size; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // CPU Multiplication
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiplication(A, B, C_cpu, width);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;

    // GPU Multiplication
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuMatrixMultiplication(A, B, C_gpu, width);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;

    // Verify results
    bool correct = true;
    for (int i = 0; i < size; i++)
    {
        if (std::abs(C_cpu[i] - C_gpu[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    printf("Matrix size: %dx%d\n", width, width);
    printf("CPU time: %.6f seconds\n", cpu_elapsed.count());
    printf("GPU time: %.6f seconds\n", gpu_elapsed.count());
    printf("Speedup: %.2fx\n", cpu_elapsed.count() / gpu_elapsed.count());
    printf("Results %s\n", correct ? "match" : "don't match");

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

    return 0;
}
