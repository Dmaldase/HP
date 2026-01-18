#include <cstdio>
#include <cuda_runtime.h>

// ===============================
// Структура параллельной очереди
// ===============================
struct Queue {
    int* data;       // массив данных
    int* head;       // индекс чтения
    int* tail;       // индекс записи
    int capacity;    // ёмкость очереди

    // Инициализация очереди
    __device__ void init(int* buffer, int* h, int* t, int size) {
        data = buffer;
        head = h;
        tail = t;
        capacity = size;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *head = 0;
            *tail = 0;
        }
    }

    // Добавление элемента в очередь
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        // переполнение
        atomicSub(tail, 1);
        return false;
    }

    // Удаление элемента из очереди
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(head, 1);
        if (pos < *tail) {
            *value = data[pos];
            return true;
        }
        // очередь пуста
        atomicSub(head, 1);
        return false;
    }
};

// ===============================
// CUDA kernel
// ===============================
__global__ void queueKernel(int* queueData, int* head, int* tail,
                            int capacity, int* results)
{
    Queue q;
    q.init(queueData, head, tail, capacity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток кладёт элемент
    q.enqueue(tid);

    __syncthreads();

    // Каждый поток пытается извлечь элемент
    int value;
    if (q.dequeue(&value)) {
        results[tid] = value;
    } else {
        results[tid] = -1;
    }
}

int main()
{
    // Количество потоков
    const int threads = 256;

    // Ёмкость очереди
    const int queueSize = 256;

    // ===============================
    // Выделение памяти
    // ===============================
    int* d_queueData;
    int* d_head;
    int* d_tail;
    int* d_results;

    cudaMalloc(&d_queueData, queueSize * sizeof(int));
    cudaMalloc(&d_head, sizeof(int));
    cudaMalloc(&d_tail, sizeof(int));
    cudaMalloc(&d_results, threads * sizeof(int));

    // ===============================
    // Запуск kernel
    // ===============================
    queueKernel<<<1, threads>>>(
        d_queueData, d_head, d_tail, queueSize, d_results
    );
    cudaDeviceSynchronize();

    // ===============================
    // Проверка корректности
    // ===============================
    int* h_results = new int[threads];
    cudaMemcpy(h_results, d_results,
               threads * sizeof(int), cudaMemcpyDeviceToHost);

    int successDequeues = 0;
    for (int i = 0; i < threads; i++) {
        if (h_results[i] != -1)
            successDequeues++;
    }

    printf("Successful dequeue operations: %d (max %d)\n",
           successDequeues, queueSize);

    // ===============================
    // Очистка
    // ===============================
    delete[] h_results;
    cudaFree(d_queueData);
    cudaFree(d_head);
    cudaFree(d_tail);
    cudaFree(d_results);

    return 0;
}
