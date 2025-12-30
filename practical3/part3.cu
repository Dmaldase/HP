#include <cuda_runtime.h>      // CUDA runtime API
#include <iostream>           // вывод на консоль
#include <algorithm>          // std::is_sorted
#include <random>             // генератор случайных чисел

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                        \
    cudaError_t e = (call);                                          \
    if (e != cudaSuccess) {                                          \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)         \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1);                                                \
    }                                                                \
} while(0)

// Количество потоков в блоке
#define BLOCK 256

// ------------------------------
// Устройство: операция siftDown
// ------------------------------
__device__ void siftDown(int* data, int start, int end)
{
    int root = start;                         // индекс корня подкучи

    while (true) {
        int child = 2 * root + 1;            // левый потомок

        if (child > end) break;              // если потомков нет — конец

        if (child + 1 <= end && data[child] < data[child + 1])
            child++;                          // выбираем большего потомка

        if (data[root] < data[child]) {
            int tmp = data[root];            // меняем root и потомка
            data[root] = data[child];
            data[child] = tmp;
            root = child;                    // продолжаем вниз
        } else {
            break;                           // свойство кучи выполнено
        }
    }
}

// ------------------------------
// CUDA kernel: параллельное построение кучи
// ------------------------------
__global__ void buildHeap(int* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // глобальный индекс

    // Все внутренние узлы начинаются с n/2 - 1 до 0
    int start = n / 2 - 1 - idx;

    if (start >= 0) {
        siftDown(data, start, n - 1);       // каждый поток heapify своего узла
    }
}

// ------------------------------
// CUDA kernel: извлечение максимума (частично параллельно)
// ------------------------------
__global__ void heapSortKernel(int* data, int n)
{
    for (int i = n - 1; i > 0; --i) {        // основной цикл извлечения
        if (threadIdx.x == 0) {
            int tmp = data[0];              // переносим максимум в конец
            data[0] = data[i];
            data[i] = tmp;
        }
        __syncthreads();                    // синхронизация потоков

        // Параллельное просеивание после удаления корня
        if (threadIdx.x == 0) {
            siftDown(data, 0, i - 1);
        }
        __syncthreads();
    }
}

// ------------------------------
// HOST main
// ------------------------------
int main()
{
    const int N = 4096;                      // размер массива

    int* h = new int[N];                    // host-массив

    std::mt19937 rng(123);                  // генератор
    std::uniform_int_distribution<int> d(0, 100000);

    for (int i = 0; i < N; i++)             // заполняем массив
        h[i] = d(rng);

    int* d_data;                            // device-массив
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h, N * sizeof(int), cudaMemcpyHostToDevice));

    // Параллельное построение кучи
    int blocks = (N / 2 + BLOCK - 1) / BLOCK;
    buildHeap<<<blocks, BLOCK>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Основная сортировка
    heapSortKernel<<<1, BLOCK>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Проверка
    if (std::is_sorted(h, h + N))
        std::cout << "Heap sort on CUDA: OK\n";
    else
        std::cout << "Heap sort FAILED\n";

    delete[] h;
    cudaFree(d_data);
}
