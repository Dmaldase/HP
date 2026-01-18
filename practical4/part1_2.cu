#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_shared(const float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    // локальный индекс потока
    unsigned int tid = threadIdx.x;

    // глобальный индекс
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // загрузка данных в shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // редукция внутри блока
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // запись результата блока
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float) * gridSize);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    reduce_shared<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // копирование частичных сумм
    float* h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);

    // финальная редукция на CPU
    float final_sum = 0.0f;
    for (int i = 0; i < gridSize; i++)
    {
        final_sum += h_output[i];
    }

    printf("Final sum = %f\n", final_sum);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
