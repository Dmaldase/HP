// Подключаем стандартную библиотеку ввода-вывода
#include <cstdio>

// Подключаем стандартную библиотеку для работы с памятью
#include <cstdlib>

// Подключаем библиотеку для замеров времени
#include <chrono>

// Подключаем CUDA Runtime API
#include <cuda_runtime.h>

// Размер блока потоков
#define BLOCK_SIZE 256

// ============================================
// Kernel 1: префиксная сумма внутри блока
// ============================================
__global__ void scanBlock(const float* input,
                          float* output,
                          float* blockSums,
                          int N)
{
    // Разделяемая память для блока
    __shared__ float temp[BLOCK_SIZE];

    // Индекс потока в блоке
    int tid = threadIdx.x;

    // Глобальный индекс элемента
    int gid = blockIdx.x * blockDim.x + tid;

    // Загрузка данных в shared memory
    if (gid < N)
        temp[tid] = input[gid];
    else
        temp[tid] = 0.0f;

    // Синхронизация потоков
    __syncthreads();

    // Inclusive scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        float val = 0.0f;

        if (tid >= offset)
            val = temp[tid - offset];

        __syncthreads();

        temp[tid] += val;

        __syncthreads();
    }

    // Запись результата
    if (gid < N)
        output[gid] = temp[tid];

    // Сохранение суммы блока
    if (tid == blockDim.x - 1)
        blockSums[blockIdx.x] = temp[tid];
} // <-- ВАЖНО: kernel ЗАКРЫТ

// ============================================
// Kernel 2: добавление сумм предыдущих блоков
// ============================================
__global__ void addBlockSums(float* data,
                             const float* prefix,
                             int N)
{
    // Глобальный индекс
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Добавляем сумму предыдущих блоков
    if (gid < N && blockIdx.x > 0)
        data[gid] += prefix[blockIdx.x - 1];
} // <-- kernel ЗАКРЫТ

// ============================================
// Последовательный scan на CPU
// ============================================
void scanCPU(const float* input, float* output, int N)
{
    output[0] = input[0];

    for (int i = 1; i < N; i++)
        output[i] = output[i - 1] + input[i];
}

// ============================================
// Главная функция
// ============================================
int main()
{
    const int N = 1'000'000;
    size_t size = N * sizeof(float);

    // Выделяем память на CPU
    float* h_input  = (float*)malloc(size);
    float* h_cpuOut = (float*)malloc(size);
    float* h_gpuOut = (float*)malloc(size);

    // Инициализация
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    // CPU scan
    auto cpuStart = std::chrono::high_resolution_clock::now();
    scanCPU(h_input, h_cpuOut, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    // GPU память
    float *d_input, *d_output, *d_blockSums;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Тайминг GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    scanBlock<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, N);

    float* h_blockPrefix = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_blockPrefix, d_blockSums,
               numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 1; i < numBlocks; i++)
        h_blockPrefix[i] += h_blockPrefix[i - 1];

    cudaMemcpy(d_blockSums, h_blockPrefix,
               numBlocks * sizeof(float), cudaMemcpyHostToDevice);

    addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_output, d_blockSums, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Проверка
    cudaMemcpy(h_gpuOut, d_output, size, cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float diff = fabs(h_cpuOut[i] - h_gpuOut[i]);
        if (diff > maxError)
            maxError = diff;
    }

    // Вывод
    printf("CPU time: %.4f ms\n", cpuTime.count());
    printf("GPU time: %.4f ms\n", gpuTime);
    printf("Max error: %f\n", maxError);

    // Очистка
    free(h_input);
    free(h_cpuOut);
    free(h_gpuOut);
    free(h_blockPrefix);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} // <-- main ЗАКРЫТ
