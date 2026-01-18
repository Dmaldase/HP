
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#define N 10000000

// Загрузка kernel из файла
std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

int main() {
    cl_int err;

    // ===============================
    // 1. Получаем платформу
    // ===============================
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // ===============================
    // 2. Получаем устройство (GPU)
    //    Для CPU заменить на CL_DEVICE_TYPE_CPU
    // ===============================
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // ===============================
    // 3. Контекст и очередь команд
    // ===============================
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // ===============================
    // 4. Данные
    // ===============================
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, A.data(), &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, B.data(), &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * N, nullptr, &err);

    // ===============================
    // 5. Компиляция ядра
    // ===============================
    std::string source = loadKernel("kernel.cl");
    const char* src = source.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);

    // ===============================
    // 6. Аргументы ядра
    // ===============================
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    size_t globalSize = N;

    // ===============================
    // 7. Запуск + замер времени
    // ===============================
    auto start = std::chrono::high_resolution_clock::now();

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;

    // ===============================
    // 8. Чтение результата
    // ===============================
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                         sizeof(float) * N, C.data(), 0, nullptr, nullptr);

    std::cout << "Execution time: " << time.count() << " ms\n";
    std::cout << "C[0] = " << C[0] << " (expected 3.0)\n";

    // ===============================
    // 9. Очистка
    // ===============================
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
