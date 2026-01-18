#include <cstdio>
#include <cuda_runtime.h>

// ===============================
// Структура параллельного стека
// ===============================
struct Stack {
    int* data;       // массив данных стека
    int* top;        // указатель вершины (в глобальной памяти)
    int capacity;    // максимальный размер

    // Инициализация стека
    __device__ void init(int* buffer, int* topPtr, int size) {
        data = buffer;
        top = topPtr;
        capacity = size;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *top = -1;
        }
    }

    // Операция push
    __device__ bool push(int value) {
        int pos = atomicAdd(top, 1) + 1;
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        // если стек переполнен
        atomicSub(top, 1);
        return false;
    }

    // Операция pop
    __device__ bool pop(int* value) {
        int pos = atomicSub(top, 1);
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        // если стек пуст
        atomicAdd(top, 1);
        return false;
    }
};

// ===============================
// CUDA kernel
// ===============================
__global__ void stackKernel(int* stackData, int* stackTop, int capacity,
                            int* popResults)
{
    Stack stack;
    stack.init(stackData, stackTop, capacity);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток кладёт своё значение
    stack.push(tid);

    __syncthreads();

    // Каждый поток пытается достать значение
    int value;
    if (stack.pop(&value)) {
        popResults[tid] = value;
    } else {
        popResults[tid] = -1;
    }
}

int main()
{
    // Количество потоков
    const int threads = 256;

    // Размер стека
    const int stackSize = 256;

    // ===============================
    // Выделение памяти
    // ===============================
    int* d_stackData;
    int* d_stackTop;
    int* d_popResults;

    cudaMalloc(&d_stackData, stackSize * sizeof(int));
    cudaMalloc(&d_stackTop, sizeof(int));
    cudaMalloc(&d_popResults, threads * sizeof(int));

    // ===============================
    // Запуск kernel
    // ===============================
    stackKernel<<<1, threads>>>(d_stackData, d_stackTop, stackSize, d_popResults);
    cudaDeviceSynchronize();

    // ===============================
    // Проверка корректности
    // ===============================
    int* h_popResults = new int[threads];
    cudaMemcpy(h_popResults, d_popResults,
               threads * sizeof(int), cudaMemcpyDeviceToHost);

    int successPops = 0;
    for (int i = 0; i < threads; i++) {
        if (h_popResults[i] != -1)
            successPops++;
    }

    printf("Successful pop operations: %d (max %d)\n",
           successPops, stackSize);

    // ===============================
    // Очистка
    // ===============================
    delete[] h_popResults;
    cudaFree(d_stackData);
    cudaFree(d_stackTop);
    cudaFree(d_popResults);

    return 0;
}
