// quicksort_cuda.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#define THREADS 256

__global__ void partitionKernel(int* in, int* out, int* flags, int n, int pivot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        flags[i] = (in[i] < pivot);
}

__global__ void scatterKernel(int* in, int* out, int* flags, int* prefix, int n, int leftCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int pos = flags[i] ? prefix[i] : (i - prefix[i] + leftCount);
        out[pos] = in[i];
    }
}

void gpuQuickSort(int* d_data, int n) {
    if (n <= 1) return;

    std::vector<int> h(n);
    cudaMemcpy(h.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    int pivot = h[n / 2];

    int* d_out;
    int* d_flags;
    int* d_prefix;

    cudaMalloc(&d_out, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_prefix, n * sizeof(int));

    int blocks = (n + THREADS - 1) / THREADS;

    partitionKernel<<<blocks, THREADS>>>(d_data, d_out, d_flags, n, pivot);

    thrust::exclusive_scan(thrust::device, d_flags, d_flags + n, d_prefix);

    int leftCount;
    cudaMemcpy(&leftCount, d_prefix + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    leftCount += h[n - 1] < pivot;

    scatterKernel<<<blocks, THREADS>>>(d_data, d_out, d_flags, d_prefix, n, leftCount);

    cudaMemcpy(d_data, d_out, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(d_flags);
    cudaFree(d_prefix);
    cudaFree(d_out);

    gpuQuickSort(d_data, leftCount);
    gpuQuickSort(d_data + leftCount, n - leftCount);
}

int main() {
    int N = 10000;
    std::vector<int> h(N);

    std::mt19937 gen;
    std::uniform_int_distribution<> dist(1, 100000);

    for (int i = 0; i < N; ++i) h[i] = dist(gen);

    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    gpuQuickSort(d, N);

    cudaMemcpy(h.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted: " << std::is_sorted(h.begin(), h.end()) << std::endl;

    cudaFree(d);
}
