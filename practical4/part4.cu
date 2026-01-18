#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

// =======================================================
// Kernel 1: редукция суммы ТОЛЬКО через global memory
// =======================================================
__global__ void reduce_global(const float* input, float* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        atomicAdd(output, input[idx]);
    }
}

// =======================================================
// Kernel 2: редукция суммы с использованием shared memory
// =======================================================
__global__ void reduce_shared(const float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(output, sdata[0]);
    }
}

int main()
{
    // размеры массивов по заданию
    int sizes[3] = {10000, 100000, 1000000};

    printf("N,Global_ms,Shared_ms\n");

    for (int s = 0; s < 3; s++)
    {
        int N = sizes[s];
        size_t size = N * sizeof(float);

        // ---------------------------
        // Host данные
        // ---------------------------
        float* h_input = (float*)malloc(size);
        for (int i = 0; i < N; i++)
            h_input[i] = 1.0f;

        // ---------------------------
        // Device данные
        // ---------------------------
        float *d_input, *d_result;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_result, sizeof(float));

        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time_global, time_shared;

        // ===================================================
        // Замер: global memory
        // ===================================================
        cudaMemset(d_result, 0, sizeof(float));

        cudaEventRecord(start);
        reduce_global<<<gridSize, BLOCK_SIZE>>>(d_input, d_result, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_global, start, stop);

        // ===================================================
        // Замер: shared memory
        // ===================================================
        cudaMemset(d_result, 0, sizeof(float));

        cudaEventRecord(start);
        reduce_shared<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_result, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_shared, start, stop);

        // ---------------------------
        // Вывод строки для графика
        // ---------------------------
        printf("%d,%.6f,%.6f\n", N, time_global, time_shared);

        // очистка
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_result);
        free(h_input);
    }

    return 0;
}
