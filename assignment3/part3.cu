// Подключаем стандартную библиотеку ввода-вывода
#include <cstdio>

// Подключаем стандартную библиотеку для malloc/free
#include <cstdlib>

// Подключаем CUDA Runtime API
#include <cuda_runtime.h>

// CUDA kernel с КОАЛЕСЦИРОВАННЫМ доступом к памяти
__global__ void coalescedKernel(const float* input, float* output, int N)
{
    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ
    if (idx < N)
    {
        // Последовательный доступ к памяти
        output[idx] = input[idx] * 2.0f;
    }
}

// CUDA kernel с НЕКОАЛЕСЦИРОВАННЫМ доступом к памяти
__global__ void nonCoalescedKernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        // каждый поток читает элемент из ДРУГОГО warp
        int warpSize = 32;
        int badIdx = (idx % warpSize) * (N / warpSize) + (idx / warpSize);

        output[idx] = input[badIdx] * 2.0f;
    }
}


int main()
{
    // Размер массива: 1 000 000 элементов
    const int N = 1 << 26;

    // Размер массива в байтах
    size_t size = N * sizeof(float);

    // Выделяем память на CPU
    float* h_input  = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Инициализация входного массива
    for (int i = 0; i < N; i++)
    {
        h_input[i] = static_cast<float>(i);
    }

    // Выделяем память на GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Копируем данные на GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Параметры запуска kernel
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // CUDA события для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =========================
    // Коалесцированный доступ
    // =========================
    cudaEventRecord(start);
    coalescedKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float coalescedTime = 0.0f;
    cudaEventElapsedTime(&coalescedTime, start, stop);

    // =========================
    // Некоалесцированный доступ
    // =========================
    cudaEventRecord(start);
    nonCoalescedKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float nonCoalescedTime = 0.0f;
    cudaEventElapsedTime(&nonCoalescedTime, start, stop);

    // Вывод результатов
    printf("Coalesced access time:     %.4f ms\n", coalescedTime);
    printf("Non-coalesced access time: %.4f ms\n", nonCoalescedTime);

    // Очистка CUDA событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Копируем результат обратно (не обязательно, но корректно)
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Освобождаем память GPU
    cudaFree(d_input);
    cudaFree(d_output);

    // Освобождаем память CPU
    free(h_input);
    free(h_output);

    return 0;
}
