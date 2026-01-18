#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// Загрузка kernel из файла
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

// Последовательное CPU-умножение для проверки
void matmul_cpu(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N, int M, int K)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

int main() {
    const int N = 128;
    const int M = 128;
    const int K = 128;

    cl_int err;

    // ===============================
    // 1. Платформа и устройство (GPU)
    // ===============================
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // ===============================
    // 2. Контекст и очередь
    // ===============================
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // ===============================
    // 3. Данные
    // ===============================
    std::vector<float> A(N * M, 1.0f);
    std::vector<float> B(M * K, 2.0f);
    std::vector<float> C_gpu(N * K, 0.0f);
    std::vector<float> C_cpu(N * K, 0.0f);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * A.size(), A.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * B.size(), B.data(), &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * C_gpu.size(), nullptr, &err);

    // ===============================
    // 4. Kernel
    // ===============================
    std::string src = loadKernel("kernel.cl");
    const char* source = src.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // ===============================
    // 5. Размеры NDRange
    // ===============================
    size_t globalSize[2] = { (size_t)N, (size_t)K };

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                           globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                        sizeof(float) * C_gpu.size(),
                        C_gpu.data(), 0, nullptr, nullptr);

    // ===============================
    // 6. Проверка на CPU
    // ===============================
    matmul_cpu(A, B, C_cpu, N, M, K);

    float maxError = 0.0f;
    for (int i = 0; i < N * K; i++) {
        maxError = std::max(maxError, std::fabs(C_cpu[i] - C_gpu[i]));
    }

    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "C[0] = " << C_gpu[0] << std::endl;

    return 0;
}
