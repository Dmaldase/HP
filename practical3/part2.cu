// =======================
// CUDA Quicksort (учебная версия)
// - Параллельное разбиение по pivot делается потоками (atomicAdd)
// - Рекурсия на GPU через Dynamic Parallelism (ядро запускает ядро)
// - Ограничение: каждый подмассив сортируется ОДНИМ блоком, n <= MAX_N
// =======================

// Подключаем runtime CUDA (cudaMalloc/cudaMemcpy/cudaDeviceSynchronize и т.д.)
#include <cuda_runtime.h>

// Подключаем ввод/вывод в консоль
#include <iostream>

// Подключаем std::is_sorted для проверки результата на CPU
#include <algorithm>

// Подключаем генератор случайных чисел (для примера)
#include <random>

// -----------------------
// Макрос для проверки ошибок CUDA вызовов
// -----------------------
#define CUDA_CHECK(call) do {                                         \
    cudaError_t e = (call);                                           \
    if (e != cudaSuccess) {                                           \
        std::cerr << "CUDA error: " << cudaGetErrorString(e)          \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
        std::exit(1);                                                 \
    }                                                                 \
} while(0)

// Количество потоков в блоке (фиксируем 256 — типичный размер)
constexpr int BLOCK_SIZE = 256;

// Максимальная длина подмассива, которую сортирует один блок (shared-память ограничена)
constexpr int MAX_N = 2048;

// Порог: если подмассив маленький — сортируем вставками (меньше накладных расходов)
constexpr int INSERTION_THRESHOLD = 32;

// -----------------------
// Вспомогательная сортировка вставками на GPU (для маленьких n)
// -----------------------
__device__ void insertion_sort_device(int* a, int n) {               // device-функция: сортирует a[0..n-1]
    for (int i = 1; i < n; ++i) {                                    // идём со 2-го элемента
        int key = a[i];                                              // сохраняем текущий элемент
        int j = i - 1;                                               // индекс слева
        while (j >= 0 && a[j] > key) {                               // пока слева элементы больше key
            a[j + 1] = a[j];                                         // сдвигаем вправо
            --j;                                                     // идём дальше влево
        }
        a[j + 1] = key;                                              // вставляем key на правильное место
    }
}

// -----------------------
// CUDA kernel: quicksort на GPU с параллельным partition
// Требование для корректности: (hi - lo + 1) <= MAX_N
// temp — глобальный временный буфер той же длины, что и основной массив
// -----------------------
__global__ void quicksort_cuda(int* a, int* temp, int lo, int hi) {   // ядро сортирует a[lo..hi]
    int n = hi - lo + 1;                                             // длина подмассива

    if (n <= 1) return;                                              // база: 0/1 элемент — уже отсортировано

    if (n <= INSERTION_THRESHOLD) {                                  // если подмассив маленький
        if (threadIdx.x == 0) {                                      // один поток
            insertion_sort_device(a + lo, n);                        // сортирует вставками
        }
        return;                                                      // выходим
    }

    if (n > MAX_N) {                                                 // если подмассив превышает лимит
        if (threadIdx.x == 0) {                                      // в учебной версии
            insertion_sort_device(a + lo, n);                        // делаем деградацию в insertion sort
        }
        return;                                                      // и выходим (чтобы не переполнить shared)
    }

    int pivot = a[hi];                                               // выбираем pivot как последний элемент

    // --- shared счётчики (общие для блока) ---
    __shared__ int lessCount;                                        // количество элементов < pivot
    __shared__ int equalCount;                                       // количество элементов == pivot
    __shared__ int greaterCount;                                     // количество элементов > pivot

    if (threadIdx.x == 0) {                                          // один поток инициализирует
        lessCount = 0;                                               // сбрасываем счётчик less
        equalCount = 0;                                              // сбрасываем счётчик equal
        greaterCount = 0;                                            // сбрасываем счётчик greater
    }
    __syncthreads();                                                 // ждём, пока счётчики обнулены

    // --- Параллельное разбиение ---
    // Каждый поток берёт элементы подмассива по шагу blockDim.x
    for (int t = threadIdx.x; t < n; t += blockDim.x) {              // t — локальный индекс в подмассиве
        int v = a[lo + t];                                           // читаем значение из глобальной памяти

        if (v < pivot) {                                             // если меньше pivot
            int pos = atomicAdd(&lessCount, 1);                      // атомарно получаем позицию в less-сегменте
            temp[lo + pos] = v;                                      // пишем в temp в начало (less)
        } else if (v > pivot) {                                      // если больше pivot
            int pos = atomicAdd(&greaterCount, 1);                   // атомарно получаем позицию в greater-сегменте
            // пока НЕ знаем старт greater-сегмента → кладём во временную область в конце,
            // но с относительным смещением от конца подмассива
            temp[hi - pos] = v;                                      // пишем "с конца" (greater)
        } else {                                                     // иначе v == pivot
            int pos = atomicAdd(&equalCount, 1);                     // атомарно получаем позицию в equal-сегменте
            // equal будем потом сдвигать в середину: пока пишем в отдельную область после less,
            // но так как lessCount меняется, мы пока пишем во вторую "служебную" область temp.
            // Для простоты: пишем equal тоже "с конца", но рядом с greater, и потом поправим.
            temp[hi - (MAX_N/2) - pos] = v;                          // резервируем под equal отдельный хвостовой диапазон
        }
    }

    __syncthreads();                                                 // ждём, пока все потоки закончили partition

    int L = lessCount;                                               // L = количество less элементов
    int G = greaterCount;                                            // G = количество greater элементов
    int E = equalCount;                                              // E = количество equal элементов

    // --- Нормализация равных pivot (перенос в середину) ---
    // Сейчас:
    // - less лежит в temp[lo .. lo+L-1]
    // - greater лежит в temp[hi-G+1 .. hi] (потому что писали с конца)
    // - equal лежит в temp[hi-(MAX_N/2)-E+1 .. hi-(MAX_N/2)]
    // Нам нужно разместить equal в temp[lo+L .. lo+L+E-1]
    if (threadIdx.x == 0) {                                          // перенос делаем одним потоком (для простоты)
        int equalSrcEnd = hi - (MAX_N/2);                            // конец области equal (включительно)
        int equalSrcStart = equalSrcEnd - E + 1;                     // начало области equal
        for (int i = 0; i < E; ++i) {                                // переносим все equal
            temp[lo + L + i] = temp[equalSrcStart + i];              // ставим equal сразу после less
        }
        // (greater уже лежит в конце подмассива temp[hi-G+1..hi], это хорошо)
    }

    __syncthreads();                                                 // ждём, пока equal перенесены

    // --- Копируем temp обратно в a в диапазоне [lo..hi] ---
    for (int t = threadIdx.x; t < n; t += blockDim.x) {              // каждый поток копирует часть диапазона
        a[lo + t] = temp[lo + t];                                    // переносим значение обратно в основной массив
    }

    __syncthreads();                                                 // гарантируем, что копирование завершено

    // --- Границы рекурсии ---
    int leftLo = lo;                                                 // левая часть начинается в lo
    int leftHi = lo + L - 1;                                         // левая часть заканчивается перед equal

    int rightLo = hi - G + 1;                                        // правая часть начинается там, где начинается greater
    int rightHi = hi;                                                // правая часть заканчивается в hi

    // --- Рекурсивные запуски на GPU (Dynamic Parallelism) ---
    // ВАЖНО: для этого нужна компиляция с -rdc=true
    if (threadIdx.x == 0) {                                          // запускает только один поток
        if (leftHi - leftLo + 1 > 1) {                               // если слева больше 1 элемента
            quicksort_cuda<<<1, BLOCK_SIZE>>>(a, temp, leftLo, leftHi); // сортируем левую часть
        }
        if (rightHi - rightLo + 1 > 1) {                             // если справа больше 1 элемента
            quicksort_cuda<<<1, BLOCK_SIZE>>>(a, temp, rightLo, rightHi); // сортируем правую часть
        }
    }
}

// -----------------------
// Host main: демонстрация
// -----------------------
int main() {                                                         // точка входа
    const int N = 2048;                                              // размер массива (должен быть <= MAX_N)
    static_assert(N <= MAX_N, "N must be <= MAX_N in this version"); // компиляторная проверка ограничения

    std::vector<int> h(N);                                           // host-массив (CPU)

    std::mt19937 rng(42);                                            // генератор случайных чисел
    std::uniform_int_distribution<int> dist(0, 100000);              // распределение значений

    for (int i = 0; i < N; ++i) {                                    // заполняем массив
        h[i] = dist(rng);                                            // случайное число
    }

    int* d_a = nullptr;                                              // указатель на device массив
    int* d_temp = nullptr;                                           // указатель на device temp буфер

    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(int)));                   // выделяем память на GPU под массив
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(int)));                // выделяем память на GPU под temp

    CUDA_CHECK(cudaMemcpy(d_a, h.data(), N * sizeof(int), cudaMemcpyHostToDevice)); // копируем h -> d_a

    quicksort_cuda<<<1, BLOCK_SIZE>>>(d_a, d_temp, 0, N - 1);         // запускаем сортировку на GPU
    CUDA_CHECK(cudaGetLastError());                                  // проверяем, что kernel запустился без ошибки
    CUDA_CHECK(cudaDeviceSynchronize());                             // ждём завершения всех kernel (включая рекурсивные)

    CUDA_CHECK(cudaMemcpy(h.data(), d_a, N * sizeof(int), cudaMemcpyDeviceToHost)); // копируем d_a -> h

    bool ok = std::is_sorted(h.begin(), h.end());                    // проверяем сортировку на CPU
    std::cout << (ok ? "OK: sorted\n" : "ERROR: not sorted\n");       // печатаем результат

    CUDA_CHECK(cudaFree(d_a));                                       // освобождаем device память массива
    CUDA_CHECK(cudaFree(d_temp));                                    // освобождаем device память temp

    return ok ? 0 : 1;                                               // код возврата (0 если всё ок)
}
