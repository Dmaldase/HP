// part4.cu
#include <cuda_runtime.h>          // базовые CUDA API: cudaMalloc/cudaMemcpy/cudaEvent...
#include <device_launch_parameters.h>
#include <iostream>               // вывод в консоль
#include <vector>                 // хранение массива на CPU
#include <random>                 // генерация случайных чисел
#include <algorithm>              // std::is_sorted (проверка)
#include <chrono>                 // замер CPU-времени (для проверки/сравнения)
#include <climits>                // INT_MAX

// -------------------------
// Утилита: проверка ошибок CUDA
// -------------------------
#define CUDA_CHECK(call) do {                                               \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";         \
        std::exit(1);                                                       \
    }                                                                       \
} while(0)

// -------------------------
// Параметры "чанка" (размера подмассива, сортируемого одним блоком)
// Важно: CHUNK должен быть степенью двойки для удобной битонной сортировки.
// -------------------------
static constexpr int CHUNK = 1024;     // один блок сортирует до 1024 элементов
static constexpr int TPB   = 256;      // threads per block (кол-во потоков в блоке)

// -------------------------
// Вспомогательная функция на GPU: compare-and-swap для битонной сортировки
// -------------------------
__device__ inline void cmpSwap(int &a, int &b, bool dir) {
    // dir=true означает сортируем по возрастанию
    // если нарушен порядок, меняем местами
    if ((a > b) == dir) {
        int tmp = a;
        a = b;
        b = tmp;
    }
}

// -------------------------
// Kernel 1: каждый блок сортирует свой CHUNK в shared memory (битонная сортировка)
// Идея: разбиваем массив на блоки по CHUNK элементов. Блок грузит свой сегмент,
// дополняет "хвост" INT_MAX, чтобы не ломать сортировку, сортирует и пишет обратно.
// -------------------------
__global__ void sortChunksBitonic(int* d, int n) {
    __shared__ int s[CHUNK];                                     // shared память на CHUNK элементов

    int blockStart = blockIdx.x * CHUNK;                         // начало сегмента (ран) для этого блока
    int tid = threadIdx.x;                                       // локальный id потока в блоке

    // Грузим данные в shared: каждый поток берет несколько элементов с шагом blockDim.x
    for (int i = tid; i < CHUNK; i += blockDim.x) {              // распределяем загрузку среди потоков
        int idx = blockStart + i;                                // индекс в глобальной памяти
        s[i] = (idx < n) ? d[idx] : INT_MAX;                     // если вышли за n, кладем INT_MAX (как "пустоту")
    }

    __syncthreads();                                             // ждем пока весь CHUNK загружен

    // Битонная сортировка CHUNK элементов в shared memory
    for (int k = 2; k <= CHUNK; k <<= 1) {                       // k = размер битонной последовательности
        for (int j = k >> 1; j > 0; j >>= 1) {                   // j = шаг сравнения в сети
            for (int i = tid; i < CHUNK; i += blockDim.x) {      // каждый поток обрабатывает свои i
                int ixj = i ^ j;                                 // "пара" для сравнения (битонная сеть)
                if (ixj > i) {                                   // чтобы не сравнивать пару дважды
                    bool dir = ((i & k) == 0);                   // направление сортировки: вверх/вниз
                    int a = s[i];                                // читаем элемент
                    int b = s[ixj];                              // читаем парный элемент
                    cmpSwap(a, b, dir);                          // при необходимости меняем
                    s[i] = a;                                    // записываем обратно
                    s[ixj] = b;                                  // записываем обратно
                }
            }
            __syncthreads();                                     // синхронизация между стадиями сети
        }
    }

    // Пишем отсортированный сегмент обратно в global memory (только реальные элементы < n)
    for (int i = tid; i < CHUNK; i += blockDim.x) {              // распределяем запись по потокам
        int idx = blockStart + i;                                // индекс в глобальной памяти
        if (idx < n) d[idx] = s[i];                              // не пишем за пределы массива
    }
}

// -------------------------
// Device-функция: merge-path поиск позиции i для k-го элемента в слиянии (A,B)
// Возвращает i (сколько берем из A), тогда j = k - i (сколько берем из B).
// Это позволяет параллельно получать элементы результата по их "рангу" k.
// -------------------------
__device__ int mergePathSearch(const int* A, int lenA, const int* B, int lenB, int k) {
    int low  = max(0, k - lenB);                                 // минимум возможного i
    int high = min(k, lenA);                                     // максимум возможного i

    while (low < high) {                                         // бинарный поиск i
        int mid = (low + high) >> 1;                             // кандидат i
        int j = k - mid;                                         // соответствующий j

        // Если A[mid] < B[j-1], значит мы взяли слишком мало из A (нужно увеличить i)
        if (mid < lenA && j > 0 && A[mid] < B[j - 1]) {
            low = mid + 1;                                       // сдвигаем нижнюю границу
        } else {
            high = mid;                                          // иначе уменьшаем верхнюю границу
        }
    }
    return low;                                                  // найденное i
}

// -------------------------
// Kernel 2: один проход слияния "ранов" длины width -> ран длины 2*width
// В каждом блоке мы сливаем одну пару: [start, start+width) и [start+width, start+2*width)
// Пишем результат в out. Источник - in.
// -------------------------
__global__ void mergePassKernel(const int* in, int* out, int n, int width) {
    int pairId = blockIdx.x;                                     // номер пары ран, которую сливает этот блок
    int start = pairId * (2 * width);                            // стартовый индекс этой пары в массиве

    if (start >= n) return;                                      // если ушли за n, ничего делать

    int lenA = min(width, n - start);                            // длина левого рана (может быть < width на конце)
    int lenB = min(width, n - (start + width));                  // длина правого рана (может быть 0 на конце)

    const int* A = in + start;                                   // указатель на левый ран
    const int* B = in + start + width;                           // указатель на правый ран
    int total = lenA + lenB;                                     // сколько элементов нужно выдать в out

    // Каждый поток делает несколько позиций k в merged массиве (k = ранги в [0..total))
    for (int k = threadIdx.x; k < total; k += blockDim.x) {      // распределяем работу по потокам
        int i = mergePathSearch(A, lenA, B, lenB, k);            // сколько взять из A для ранга k
        int j = k - i;                                           // сколько взять из B

        int aVal = (i < lenA) ? A[i] : INT_MAX;                  // кандидат из A (или бесконечность)
        int bVal = (j < lenB) ? B[j] : INT_MAX;                  // кандидат из B (или бесконечность)

        out[start + k] = (aVal <= bVal) ? aVal : bVal;           // k-й элемент merged = min(aVal, bVal)
    }
}

// -------------------------
// GPU merge sort: 
// 1) сортировка чанков CHUNK (каждый блок) -> получаем отсортированные ранки CHUNK
// 2) итеративные merge-проходы: width=CHUNK, 2*CHUNK, 4*CHUNK... пока width>=n
// -------------------------
float gpuMergeSort(int* d_data, int n) {
    // Выделяем второй буфер для "пинг-понга" на merge-проходах
    int* d_tmp = nullptr;                                        // временный массив для результата merge-pass
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(int)));             // аллокация памяти на GPU

    // 1) Сортируем чанки
    int numBlocksChunks = (n + CHUNK - 1) / CHUNK;               // сколько чанков по CHUNK
    sortChunksBitonic<<<numBlocksChunks, TPB>>>(d_data, n);      // запускаем kernel сортировки чанков
    CUDA_CHECK(cudaGetLastError());                              // проверяем запуск kernel
    CUDA_CHECK(cudaDeviceSynchronize());                         // ждем завершения (чтобы корректно мерить merge отдельно, если надо)

    // 2) Замеряем время всех merge-проходов (события CUDA измеряют время на GPU)
    cudaEvent_t evStart, evStop;                                 // CUDA events для тайминга
    CUDA_CHECK(cudaEventCreate(&evStart));                       // создаем событие старта
    CUDA_CHECK(cudaEventCreate(&evStop));                        // создаем событие стопа
    CUDA_CHECK(cudaEventRecord(evStart));                        // ставим отметку "старт"

    const int* in = d_data;                                      // текущий вход для merge-pass
    int* out = d_tmp;                                            // текущий выход для merge-pass

    for (int width = CHUNK; width < n; width <<= 1) {            // увеличиваем длину ранков в 2 раза каждый проход
        int numPairs = (n + (2 * width) - 1) / (2 * width);      // сколько пар ранков нужно слить
        mergePassKernel<<<numPairs, TPB>>>(in, out, n, width);   // запускаем слияние пар ранков
        CUDA_CHECK(cudaGetLastError());                          // проверяем kernel launch

        // Меняем местами вход и выход (ping-pong), чтобы не делать лишние копирования
        const int* newIn = out;                                  // теперь результат станет входом
        out = (out == d_tmp) ? (int*)d_data : d_tmp;             // чередуем буферы
        in = newIn;                                              // обновляем вход
    }

    // Если итог оказался не в d_data, копируем обратно (это происходит, когда число проходов нечетное)
    if (in != d_data) {                                          // если финальные данные в d_tmp
        CUDA_CHECK(cudaMemcpy(d_data, in, n * sizeof(int), cudaMemcpyDeviceToDevice)); // копируем внутри GPU
    }

    CUDA_CHECK(cudaEventRecord(evStop));                         // ставим отметку "стоп"
    CUDA_CHECK(cudaEventSynchronize(evStop));                    // ждем, пока GPU закончит работу до evStop

    float ms = 0.0f;                                             // сюда запишем миллисекунды
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));      // считаем время между событиями

    CUDA_CHECK(cudaEventDestroy(evStart));                       // освобождаем ресурс события
    CUDA_CHECK(cudaEventDestroy(evStop));                        // освобождаем ресурс события
    CUDA_CHECK(cudaFree(d_tmp));                                 // освобождаем временный буфер

    return ms;                                                   // возвращаем время сортировки на GPU (ms)
}

// -------------------------
// CPU сортировка для проверки корректности (и грубого сравнения)
// -------------------------
double cpuSortMs(std::vector<int> a) {
    auto t0 = std::chrono::high_resolution_clock::now();         // старт таймера CPU
    std::sort(a.begin(), a.end());                               // сортировка на CPU
    auto t1 = std::chrono::high_resolution_clock::now();         // стоп таймера CPU
    return std::chrono::duration<double, std::milli>(t1 - t0).count(); // время в ms
}

int main() {
    // Тестовые размеры по заданию
    std::vector<int> sizes = {10000, 100000};                    // N=10k и N=100k

    // Генератор случайных данных (один и тот же подход для обеих размерностей)
    std::mt19937 gen(std::random_device{}());                    // PRNG
    std::uniform_int_distribution<> dist(1, 1000000);            // значения 1..1e6

    for (int n : sizes) {
        // Готовим входной массив на CPU
        std::vector<int> h(n);                                   // host массив
        for (int i = 0; i < n; ++i) h[i] = dist(gen);            // заполняем случайными числами

        // Копия для CPU-сортировки (чтобы сравнивать по одинаковым данным)
        std::vector<int> h_cpu = h;                              // копируем исходные данные

        // Выделяем память на GPU и копируем вход
        int* d = nullptr;                                        // device указатель
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));             // аллокация device памяти
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(int), cudaMemcpyHostToDevice)); // H2D копирование

        // GPU сортировка + замер
        float gpuMs = gpuMergeSort(d, n);                        // сортируем на GPU и получаем время

        // Возвращаем результат на CPU и проверяем отсортированность
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost)); // D2H копирование
        bool ok = std::is_sorted(h.begin(), h.end());            // проверка, что массив неубывает

        // CPU сортировка для сравнения (не требуется заданием, но полезно для отчета)
        double cpuMs = cpuSortMs(h_cpu);                         // сортируем на CPU и мерим время

        // Печать результатов
        std::cout << "N = " << n << "\n";                        // печать размера
        std::cout << "GPU merge-sort time: " << gpuMs << " ms\n";// время GPU
        std::cout << "CPU std::sort time:  " << cpuMs << " ms\n";// время CPU
        std::cout << "Sorted correct: " << (ok ? "YES" : "NO") << "\n\n"; // корректность

        CUDA_CHECK(cudaFree(d));                                 // освобождаем GPU память
    }

    return 0;                                                    // завершение программы
}
