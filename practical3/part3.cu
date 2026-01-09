#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static inline int floor_log2_u32(unsigned x) {
    int r = 0;
    while (x >>= 1) ++r;
    return r;
}

__device__ __forceinline__ void siftDown(int* a, int n, int idx) {
    while (true) {
        int left = 2 * idx + 1;
        if (left >= n) break;
        int right = left + 1;

        int largest = idx;
        int lv = a[left];
        int av = a[largest];
        if (lv > av) largest = left;

        if (right < n) {
            int rv = a[right];
            av = a[largest];
            if (rv > av) largest = right;
        }

        if (largest == idx) break;

        int tmp = a[idx];
        a[idx] = a[largest];
        a[largest] = tmp;

        idx = largest;
    }
}

__global__ void heapifyLevelKernel(int* a, int n, int levelStart, int levelCount) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= levelCount) return;
    int idx = levelStart + t;
    siftDown(a, n, idx);
}

__global__ void swapRootWithEndKernel(int* a, int endIdx) {
    int tmp = a[0];
    a[0] = a[endIdx];
    a[endIdx] = tmp;
}

__global__ void siftDownRootKernel(int* a, int heapSize) {
    siftDown(a, heapSize, 0);
}

void gpuHeapSort(int* d_a, int n) {
    if (n <= 1) return;

    int lastInternal = (n / 2) - 1;
    if (lastInternal >= 0) {
        int lastInternalDepth = floor_log2_u32((unsigned)(lastInternal + 1));
        constexpr int THREADS = 256;

        for (int depth = lastInternalDepth; depth >= 0; --depth) {
            int levelStart = (1 << depth) - 1;
            int levelEnd = (1 << (depth + 1)) - 2;
            if (levelStart > lastInternal) continue;
            if (levelEnd > lastInternal) levelEnd = lastInternal;

            int levelCount = levelEnd - levelStart + 1;
            int blocks = (levelCount + THREADS - 1) / THREADS;

            heapifyLevelKernel<<<blocks, THREADS>>>(d_a, n, levelStart, levelCount);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    for (int end = n - 1; end >= 1; --end) {
        swapRootWithEndKernel<<<1, 1>>>(d_a, end);
        CUDA_CHECK(cudaGetLastError());

        siftDownRootKernel<<<1, 1>>>(d_a, end);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int main() {
    const int N = 1000000;
    std::vector<int> h(N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-1000000, 1000000);
    for (int i = 0; i < N; ++i) h[i] = dist(rng);

    int* d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    gpuHeapSort(d_a, N);

    CUDA_CHECK(cudaMemcpy(h.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));

    if (!std::is_sorted(h.begin(), h.end())) {
        std::cerr << "Sort FAILED\n";
        return 1;
    }

    std::cout << "Sort OK\n";
    return 0;
}
