#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float *A, float *B, float *C, int width)
{
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if within matrix bounds
    if (row < width && col < width)
    {
        float sum = 0.0f;
        // Compute dot product of row of A and column of B
        for (int k = 0; k < width; k++)
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Host function for matrix multiplication
void matrixMultiplication(float *A, float *B, float *C, int width)
{
    int size = width * width * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (width + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    // Matrix dimensions
    int width = 1024;
    int size = width * width;

    // Allocate host memory
    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];

    // Initialize matrices
    for (int i = 0; i < size; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Perform matrix multiplication
    matrixMultiplication(A, B, C, width);

    // Verify result (check just a few values)
    printf("C[0][0] = %.2f (expected: %.2f)\n", C[0], width * 2.0f);
    printf("C[10][10] = %.2f (expected: %.2f)\n", C[10 * width + 10], width * 2.0f);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
