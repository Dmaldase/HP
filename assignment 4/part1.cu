// Подключаем стандартную библиотеку ввода-вывода
#include <cstdio>

// Подключаем стандартную библиотеку для работы с памятью
#include <cstdlib>

// Подключаем библиотеку для замеров времени на CPU
#include <chrono>

// Подключаем CUDA Runtime API
#include <cuda_runtime.h>

// ==========================================
// CUDA kernel: суммирование через global memory
// ==========================================
__global__ void sumKernel(const float* data, float* result, int N)
{
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за границы массива
    if (idx < N) {
        // Атомарно прибавляем элемент массива к общей сумме
        // Используется глобальная память
        atomicAdd(result, data[idx]);
    }
}

// ==========================================
// Последовательная реализация на CPU
// ==========================================
float sumCPU(const float* data, int N)
{
    // Переменная для хранения суммы
    float sum = 0.0f;

    // Последовательный проход по массиву
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }

    // Возвращаем результат
    return sum;
}

int main()
{
    // Размер массива
    const int N = 100000;

    // Размер массива в байтах
    size_t size = N * sizeof(float);

    // ==========================================
    // Выделение памяти на CPU
    // ==========================================
    float* h_data = (float*)malloc(size);

    // Инициализация массива (все элементы равны 1)
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // ==========================================
    // Вычисление суммы на CPU + замер времени
    // ==========================================
    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Вызов последовательной функции
    float cpuSum = sumCPU(h_data, N);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    // Вычисляем время выполнения на CPU
    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    // ==========================================
    // Выделение памяти на GPU
    // ==========================================
    float* d_data;    // массив на GPU
    float* d_result;  // переменная для результата

    cudaMalloc(&d_data, size);
    cudaMalloc(&d_result, sizeof(float));

    // Копируем входной массив с CPU на GPU
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Обнуляем результат на GPU
    cudaMemset(d_result, 0, sizeof(float));

    // ==========================================
    // Настройка конфигурации запуска kernel
    // ==========================================
    int blockSize = 256;                          // число потоков в блоке
    int gridSize  = (N + blockSize - 1) / blockSize; // число блоков

    // ==========================================
    // Замер времени выполнения на GPU
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запоминаем время начала
    cudaEventRecord(start);

    // Запуск CUDA kernel
    sumKernel<<<gridSize, blockSize>>>(d_data, d_result, N);

    // Запоминаем время окончания
    cudaEventRecord(stop);

    // Ждём завершения kernel
    cudaEventSynchronize(stop);

    // Переменная для хранения времени GPU
    float gpuTime = 0.0f;

    // Вычисляем время выполнения kernel
    cudaEventElapsedTime(&gpuTime, start, stop);

    // ==========================================
    // Копирование результата с GPU на CPU
    // ==========================================
    float gpuSum = 0.0f;
    cudaMemcpy(&gpuSum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // ==========================================
    // Вывод результатов
    // ==========================================
    printf("CPU sum: %f\n", cpuSum);
    printf("GPU sum: %f\n", gpuSum);
    printf("CPU time: %.4f ms\n", cpuTime.count());
    printf("GPU time: %.4f ms\n", gpuTime);

    // ==========================================
    // Освобождение ресурсов
    // ==========================================
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Завершение программы
    return 0;
}
