#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// Безопасные значения почти для любой CUDA-видеокарты
#define CHUNK_SIZE 256
#define THREADS    256

// Печать ошибки с КОДОМ и текстом
#define CUDA_CHECK(call) do {                                                \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
        std::cerr << "CUDA error code=" << (int)e                            \
                  << " msg=" << cudaGetErrorString(e)                        \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
        std::exit(1);                                                        \
    }                                                                        \
} while(0)

__device__ __forceinline__ void dswap(int& a, int& b) {
    int t = a; a = b; b = t;
}

// Каждый блок сортирует свой чанк odd-even сортировкой
__global__ void sortChunks(int* data, int n) {

    __shared__ int buf[CHUNK_SIZE];

    int blockStart = (int)blockIdx.x * CHUNK_SIZE;
    int tid = (int)threadIdx.x;

    int blockSize = n - blockStart;
    if (blockSize > CHUNK_SIZE) blockSize = CHUNK_SIZE;
    if (blockSize <= 0) return;

    if (tid < blockSize)
        buf[tid] = data[blockStart + tid];

    __syncthreads();

    for (int step = 0; step < blockSize; ++step) {
        if (((tid & 1) == (step & 1)) && (tid + 1 < blockSize)) {
            if (buf[tid] > buf[tid + 1]) dswap(buf[tid], buf[tid + 1]);
        }
        __syncthreads();
    }

    if (tid < blockSize)
        data[blockStart + tid] = buf[tid];
}

// CPU merge: right exclusive
void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> temp(right - left);
    int i = left, j = mid, k = 0;

    while (i < mid && j < right)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];

    while (i < mid) temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];

    std::copy(temp.begin(), temp.end(), arr.begin() + left);
}

int main() {
    // 1) Проверяем, что CUDA вообще видит устройство
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "CUDA devices not found (devCount=0)\n";
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    // 2) Печатаем базовую инфу по GPU (чтобы понимать ограничения)
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name
              << "  maxThreadsPerBlock=" << prop.maxThreadsPerBlock
              << "  sharedPerBlock=" << prop.sharedMemPerBlock << "\n";

    const int N = 1 << 20; // ~1,048,576

    std::vector<int> arr(N);
    std::mt19937 gen(0);
    for (int& x : arr) x = (int)gen();

    int* d_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_arr, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, arr.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    sortChunks<<<blocks, THREADS>>>(d_arr, N);

    // 3) Ловим ошибку запуска ядра
    CUDA_CHECK(cudaGetLastError());

    // 4) Ловим runtime-ошибки выполнения ядра
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(arr.data(), d_arr, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

    // Диагностика: проверим, что каждый чанк отсортирован
    for (int b = 0; b < blocks; ++b) {
        int L = b * CHUNK_SIZE;
        int R = std::min(L + CHUNK_SIZE, N);
        if (!std::is_sorted(arr.begin() + L, arr.begin() + R)) {
            std::cout << "Chunk sort FAILED at block " << b << "\n";
            CUDA_CHECK(cudaFree(d_arr));
            return 1;
        }
    }

    // CPU merge
    for (int size = CHUNK_SIZE; size < N; size *= 2) {
        for (int left = 0; left < N; left += 2 * size) {
            int mid = std::min(left + size, N);
            int right = std::min(left + 2 * size, N);
            merge(arr, left, mid, right);
        }
    }

    std::cout << (std::is_sorted(arr.begin(), arr.end()) ? "Sorting OK\n" : "Sorting FAILED\n");

    CUDA_CHECK(cudaFree(d_arr));
    return 0;
}
