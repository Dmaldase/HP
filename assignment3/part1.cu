// Подключаем основной заголовок CUDA Runtime API.
// Без него не будет cudaMalloc, cudaMemcpy, cudaEvent и прочих радостей жизни.
#include <cuda_runtime.h>

// Стандартный ввод-вывод C++ для std::cout
#include <iostream>

// Вектор из стандартной библиотеки C++ для хранения данных на CPU
#include <vector>

// Размер массива: 1 000 000 элементов
#define N 1000000

// Количество потоков в одном блоке CUDA
#define THREADS 256

// Число, на которое будем умножать каждый элемент массива
#define MULTIPLIER 3.14f

// =====================================================
// KERNEL 1: Использование только глобальной памяти
// =====================================================

// __global__ означает, что функция:
// 1) вызывается с CPU
// 2) выполняется на GPU
__global__ void multiply_global(float* data, float value, int n) {

    // Глобальный индекс потока:
    // blockIdx.x — номер блока
    // blockDim.x — число потоков в блоке
    // threadIdx.x — номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка, чтобы не выйти за пределы массива
    if (idx < n) {
        // Каждый поток умножает ОДИН элемент массива
        // Чтение и запись происходят напрямую в глобальной памяти GPU
        data[idx] *= value;
    }
}

// =====================================================
// KERNEL 2: Использование shared memory
// =====================================================

// Shared memory работает быстрее глобальной,
// но доступна только внутри блока
__global__ void multiply_shared(float* data, float value, int n) {

    // Объявляем разделяемую память.
    // Размер равен количеству потоков в блоке
    __shared__ float cache[THREADS];

    // Глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Локальный индекс потока внутри блока
    int tid = threadIdx.x;

    // Загружаем данные из глобальной памяти в shared memory
    if (idx < n)
        cache[tid] = data[idx];
    else
        // Если поток "лишний", кладём 0, чтобы не было мусора
        cache[tid] = 0.0f;

    // Синхронизация потоков:
    // все потоки блока должны закончить загрузку
    __syncthreads();

    // Выполняем вычисление в shared memory
    cache[tid] *= value;

    // Снова синхронизация,
    // чтобы все потоки закончили вычисления
    __syncthreads();

    // Записываем результат обратно в глобальную память
    if (idx < n)
        data[idx] = cache[tid];
}

int main() {

    // Размер памяти в байтах
    size_t size = N * sizeof(float);

    // =======================
    // HOST (CPU)
    // =======================

    // Создаём массив на CPU и инициализируем единицами
    std::vector<float> h_data(N, 1.0f);

    // =======================
    // DEVICE (GPU)
    // =======================

    // Указатель на память GPU
    float* d_data;

    // Выделяем память на GPU
    cudaMalloc(&d_data, size);

    // =======================
    // Таймеры CUDA
    // =======================

    // События для измерения времени
    cudaEvent_t start, stop;

    // Создаём события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Вычисляем количество блоков
    // (округление вверх)
    int blocks = (N + THREADS - 1) / THREADS;

    // =======================
    // Запуск версии с глобальной памятью
    // =======================

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    // Запускаем таймер
    cudaEventRecord(start);

    // Запуск kernel:
    // <<<количество блоков, потоков в блоке>>>
    multiply_global<<<blocks, THREADS>>>(d_data, MULTIPLIER, N);

    // Останавливаем таймер
    cudaEventRecord(stop);

    // Ждём завершения kernel
    cudaEventSynchronize(stop);

    // Переменная для времени выполнения
    float time_global;

    // Вычисляем время в миллисекундах
    cudaEventElapsedTime(&time_global, start, stop);

    // =======================
    // Запуск версии с shared memory
    // =======================

    // Снова копируем исходные данные на GPU
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    // Запускаем таймер
    cudaEventRecord(start);

    // Запуск kernel с shared memory
    multiply_shared<<<blocks, THREADS>>>(d_data, MULTIPLIER, N);

    // Останавливаем таймер
    cudaEventRecord(stop);

    // Ждём завершения
    cudaEventSynchronize(stop);

    // Переменная для времени
    float time_shared;

    // Получаем время выполнения
    cudaEventElapsedTime(&time_shared, start, stop);

    // =======================
    // Вывод результатов
    // =======================

    std::cout << "Array size: " << N << std::endl;
    std::cout << "Global memory time: " << time_global << " ms" << std::endl;
    std::cout << "Shared memory time: " << time_shared << " ms" << std::endl;

    // Освобождаем память GPU
    cudaFree(d_data);

    // Уничтожаем события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Успешное завершение программы
    return 0;
}
