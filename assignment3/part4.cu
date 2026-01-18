#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel: поэлементное сложение массивов
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // Размер массива (достаточно большой)
    const int N = 1 << 24; // ~16 млн элементов
    size_t size = N * sizeof(float);

    // Память на CPU
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Память на GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Варианты размеров блока
    int blockSizes[] = {64, 128, 256, 512, 1024};
    int numTests = 5;

    // CUDA события
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BlockSize | Time (ms)\n");
    printf("---------------------\n");

    for (int i = 0; i < numTests; i++)
    {
        int blockSize = blockSizes[i];
        int gridSize = (N + blockSize - 1) / blockSize;

        // warm-up
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        // замер
        cudaEventRecord(start);
        for (int iter = 0; iter < 100; iter++)
        {
            vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs;
        cudaEventElapsedTime(&timeMs, start, stop);

        timeMs /= 100.0f; // среднее время одного запуска

        printf("%8d | %8.4f\n", blockSize, timeMs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
