// Подключаем стандартную библиотеку ввода-вывода для printf
#include <cstdio>

// Подключаем стандартную библиотеку для работы с памятью и rand()
#include <cstdlib>

// Подключаем chrono для измерения времени на CPU
#include <chrono>

// Подключаем CUDA Runtime API
#include <cuda_runtime.h>

// CUDA kernel: поэлементное сложение массивов A и B, результат в C
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за пределы массива
    if (idx < N)
    {
        // Поэлементно складываем два массива
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // Размер массива (можно менять для экспериментов)
    const int N = 1 << 24; // ~16 миллионов элементов

    // Размер массива в байтах
    size_t size = N * sizeof(float);

    // Выделяем память на хосте (CPU)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Инициализируем входные массивы случайными значениями
    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Выделяем память на устройстве (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Набор размеров блока потоков для экспериментов
    int blockSizes[] = {128, 256, 512};

    // Проходимся по каждому размеру блока
    for (int b = 0; b < 3; b++)
    {
        int blockSize = blockSizes[b];

        // Вычисляем количество блоков
        int gridSize = (N + blockSize - 1) / blockSize;

        // Создаём CUDA события для замера времени
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Записываем момент начала выполнения kernel
        cudaEventRecord(start);

        // Запуск CUDA kernel
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

        // Записываем момент окончания выполнения kernel
        cudaEventRecord(stop);

        // Ждём завершения kernel
        cudaEventSynchronize(stop);

        // Переменная для хранения времени выполнения
        float milliseconds = 0.0f;

        // Вычисляем время между start и stop
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Выводим результат измерения
        printf("Block size: %d, Time: %.4f ms\n", blockSize, milliseconds);

        // Освобождаем CUDA события
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Копируем результат обратно с GPU на CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Проверка корректности результата
    bool correct = true;
    for (int i = 0; i < N; i++)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    // Выводим результат проверки
    if (correct)
        printf("Result is correct.\n");
    else
        printf("Result is incorrect!\n");

    // Освобождаем память на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Освобождаем память на CPU
    free(h_A);
    free(h_B);
    free(h_C);

    // Завершаем программу
    return 0;
}
