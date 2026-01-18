#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N 1024
#define CHUNK_SIZE 32


__global__ void bubble_sort_chunks(float* data)
{
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    int start = chunk_id * CHUNK_SIZE;

    if (start >= N) return;

    // ЛОКАЛЬНАЯ память (регистры потока)
    float local[CHUNK_SIZE];

    // загрузка из global memory
    for (int i = 0; i < CHUNK_SIZE; i++)
    {
        local[i] = data[start + i];
    }

    // пузырьковая сортировка
    for (int i = 0; i < CHUNK_SIZE - 1; i++)
    {
        for (int j = 0; j < CHUNK_SIZE - i - 1; j++)
        {
            if (local[j] > local[j + 1])
            {
                float tmp = local[j];
                local[j] = local[j + 1];
                local[j + 1] = tmp;
            }
        }
    }

    // запись обратно в global memory
    for (int i = 0; i < CHUNK_SIZE; i++)
    {
        data[start + i] = local[i];
    }
}

__global__ void merge_chunks(float* input, float* output)
{
    __shared__ float left[CHUNK_SIZE];
    __shared__ float right[CHUNK_SIZE];

    int pair_id = blockIdx.x;
    int left_start = pair_id * 2 * CHUNK_SIZE;
    int right_start = left_start + CHUNK_SIZE;

    if (right_start >= N) return;

    // загрузка в shared memory
    for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x)
    {
        left[i]  = input[left_start + i];
        right[i] = input[right_start + i];
    }
    __syncthreads();

    // один поток делает слияние (упрощение для учебной задачи)
    if (threadIdx.x == 0)
    {
        int i = 0, j = 0, k = 0;
        while (i < CHUNK_SIZE && j < CHUNK_SIZE)
        {
            if (left[i] < right[j])
                output[left_start + k++] = left[i++];
            else
                output[left_start + k++] = right[j++];
        }
        while (i < CHUNK_SIZE)
            output[left_start + k++] = left[i++];
        while (j < CHUNK_SIZE)
            output[left_start + k++] = right[j++];
    }
}

int main()
{
    float h_data[N];
    for (int i = 0; i < N; i++)
        h_data[i] = static_cast<float>(rand() % 100);

    float *d_data, *d_tmp;
    cudaMalloc(&d_data, sizeof(float) * N);
    cudaMalloc(&d_tmp, sizeof(float) * N);

    cudaMemcpy(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice);

    int threads = 128;
    int chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // сортировка подмассивов
    bubble_sort_chunks<<<(chunks + threads - 1) / threads, threads>>>(d_data);
    cudaDeviceSynchronize();

    // слияние
    merge_chunks<<<chunks / 2, 32>>>(d_data, d_tmp);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_tmp, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("Sorted result (first 32 elements):\n");
    for (int i = 0; i < 32; i++)
        printf("%f ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_tmp);
    return 0;
}

