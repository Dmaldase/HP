#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(x) do { if((x)!=cudaSuccess){ \
    std::cout<<"CUDA error\n"; exit(1);} } while(0)

// ================= GPU HEAPSORT (from part 3) =================

__device__ __forceinline__ void siftDown(int* a, int n, int idx) {
    while (true) {
        int left = 2 * idx + 1;
        if (left >= n) break;
        int right = left + 1;
        int largest = (right < n && a[right] > a[left]) ? right : left;
        if (a[idx] >= a[largest]) break;
        int t = a[idx]; a[idx] = a[largest]; a[largest] = t;
        idx = largest;
    }
}

__global__ void heapifyKernel(int* a, int n, int start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < n / 2) siftDown(a, n, i);
}

__global__ void swapKernel(int* a, int end) {
    int t = a[0]; a[0] = a[end]; a[end] = t;
}

__global__ void fixRoot(int* a, int n) {
    siftDown(a, n, 0);
}

void gpuHeapSort(int* d, int n) {
    int threads = 256;
    int blocks = (n/2 + threads - 1) / threads;
    heapifyKernel<<<blocks, threads>>>(d, n, 0);
    cudaDeviceSynchronize();

    for (int end = n - 1; end > 0; --end) {
        swapKernel<<<1,1>>>(d, end);
        fixRoot<<<1,1>>>(d, end);
        cudaDeviceSynchronize();
    }
}

// ================= CPU HEAPSORT =================

void cpuHeapSort(std::vector<int>& a) {
    std::make_heap(a.begin(), a.end());
    std::sort_heap(a.begin(), a.end());
}

// ================= BENCHMARK =================

void runTest(int N) {
    std::vector<int> data(N);
    std::mt19937 rng(42);
    for (int& x : data) x = rng();

    std::vector<int> cpu = data;

    auto c1 = std::chrono::high_resolution_clock::now();
    cpuHeapSort(cpu);
    auto c2 = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(c2 - c1).count();

    int* d;
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t g1, g2;
    cudaEventCreate(&g1); cudaEventCreate(&g2);

    cudaEventRecord(g1);
    gpuHeapSort(d, N);
    cudaEventRecord(g2);
    cudaEventSynchronize(g2);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, g1, g2);

    cudaMemcpy(data.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    std::cout << "N = " << N 
              << " | CPU: " << cpuTime << " ms"
              << " | GPU: " << gpuTime << " ms"
              << " | Speedup: " << cpuTime / gpuTime << "x\n";
}

int main() {
    runTest(10000);
    runTest(100000);
    runTest(1000000);
}
