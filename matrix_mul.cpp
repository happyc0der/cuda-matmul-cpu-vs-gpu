#include <iostream>
#include <chrono>

// Function to perform matrix multiplication
void matrixMultiplication(float *A, float *B, float *C, int width)
{
    // Compute each element of the result matrix
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
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
}

int main()
{
    // Matrix dimensions - same as CUDA example
    int width = 1024;
    int size = width * width;

    // Allocate host memory - same as CUDA example
    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];

    // Initialize matrices with the same values as CUDA example
    for (int i = 0; i < size; i++)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matrixMultiplication(A, B, C, width);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Verify result (check just a few values) - same as CUDA example
    printf("C[0][0] = %.2f (expected: %.2f)\n", C[0], width * 2.0f);
    printf("C[10][10] = %.2f (expected: %.2f)\n", C[10 * width + 10], width * 2.0f);
    printf("Execution time: %.2f seconds\n", elapsed.count());

    // Free host memory - same as CUDA example
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
