// merge_sort_cuda.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#define CHUNK 1024
#define THREADS 256

__global__ void bitonicSortBlock(int* data, int n) {
    __shared__ int s[CHUNK];

    int start = blockIdx.x * CHUNK;
    int tid = threadIdx.x;

    if (start + tid < n)
        s[tid] = data[start + tid];
    else
        s[tid] = INT_MAX;

    __syncthreads();

    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int i = tid;
            int ixj = i ^ j;
            if (ixj > i && ixj < CHUNK) {
                if ((s[i] > s[ixj]) == ((i & k) == 0)) {
                    int tmp = s[i];
                    s[i] = s[ixj];
                    s[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (start + tid < n)
        data[start + tid] = s[tid];
}

__global__ void mergePass(int* in, int* out, int n, int width) {
    int start = blockIdx.x * 2 * width;
    int i = start;
    int j = start + width;
    int end1 = min(start + width, n);
    int end2 = min(start + 2 * width, n);

    int k = start + threadIdx.x;
    while (k < end2) {
        int a = (i < end1) ? in[i] : INT_MAX;
        int b = (j < end2) ? in[j] : INT_MAX;
        out[k] = (a < b) ? (i++, a) : (j++, b);
        k += blockDim.x;
    }
}

void gpuMergeSort(int* d, int n) {
    int* temp;
    cudaMalloc(&temp, n * sizeof(int));

    int blocks = (n + CHUNK - 1) / CHUNK;
    bitonicSortBlock<<<blocks, CHUNK>>>(d, n);

    for (int width = CHUNK; width < n; width *= 2) {
        int pairs = (n + 2 * width - 1) / (2 * width);
        mergePass<<<pairs, THREADS>>>(d, temp, n, width);
        std::swap(d, temp);
    }

    cudaFree(temp);
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

    gpuMergeSort(d, N);

    cudaMemcpy(h.data(), d, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted: " << std::is_sorted(h.begin(), h.end()) << std::endl;

    cudaFree(d);
}
