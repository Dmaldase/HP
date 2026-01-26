// Подключаем стандартную библиотеку ввода-вывода
#include <cstdio>

// Подключаем стандартную библиотеку для работы с памятью
#include <cstdlib>

// Подключаем библиотеку для замера времени
#include <chrono>

// Подключаем CUDA Runtime API
#include <cuda_runtime.h>

// ==========================================
// CUDA kernel: обработка части массива
// ==========================================
__global__ void processGPU(const float* input,
                           float* output,
                           int start,
                           int end)
{
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;

    // Проверка границ
    if (idx < end) {
        // Простая операция обработки
        output[idx] = input[idx] * 2.0f;
    }
}

// ==========================================
// CPU обработка массива
// ==========================================
void processCPU(const float* input,
                float* output,
                int start,
                int end)
{
    // Последовательная обработка диапазона
    for (int i = start; i < end; i++) {
        output[i] = input[i] * 2.0f;
    }
}

int main()
{
    // Размер массива
    const int N = 1'000'000;

    // Размер массива в байтах
    size_t size = N * sizeof(float);

    // ==========================================
    // Выделение памяти на CPU
    // ==========================================
    float* h_input  = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Инициализация входного массива
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // ==========================================
    // 1. Только CPU
    // ==========================================
    auto cpuStart = std::chrono::high_resolution_clock::now();

    processCPU(h_input, h_output, 0, N);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    // ==========================================
    // Выделение памяти на GPU
    // ==========================================
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // ==========================================
    // 2. Только GPU
    // ==========================================
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    processGPU<<<gridSize, blockSize>>>(d_input, d_output, 0, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // ==========================================
    // 3. Гибрид: CPU + GPU
    // ==========================================
    int mid = N / 2;

    // CPU часть
    auto hybridCpuStart = std::chrono::high_resolution_clock::now();
    processCPU(h_input, h_output, 0, mid);
    auto hybridCpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> hybridCpuTime =
        hybridCpuEnd - hybridCpuStart;

    // GPU часть
    cudaEventRecord(start);

    processGPU<<<gridSize, blockSize>>>(d_input, d_output, mid, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float hybridGpuTime = 0.0f;
    cudaEventElapsedTime(&hybridGpuTime, start, stop);

    float hybridTotalTime = hybridCpuTime.count() + hybridGpuTime;

    // ==========================================
    // Копирование результата GPU
    // ==========================================
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // ==========================================
    // Вывод результатов
    // ==========================================
    printf("CPU time:     %.4f ms\n", cpuTime.count());
    printf("GPU time:     %.4f ms\n", gpuTime);
    printf("Hybrid time:  %.4f ms\n", hybridTotalTime);

    // ==========================================
    // Очистка памяти
    // ==========================================
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
