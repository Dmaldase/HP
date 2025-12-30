// Подключаем CUDA runtime API (cudaMalloc, cudaMemcpy, cudaDeviceSynchronize и т.д.)
#include <cuda_runtime.h>

// Подключаем потоковый вывод в консоль
#include <iostream>

// Подключаем контейнер vector для удобной работы с массивами на CPU
#include <vector>

// Подключаем алгоритмы стандартной библиотеки (sort, make_heap, sort_heap, is_sorted)
#include <algorithm>

// Подключаем генератор случайных чисел
#include <random>

// Подключаем библиотеку для измерения времени
#include <chrono>

// Даём псевдоним типу таймера для удобства
using clk = std::chrono::high_resolution_clock;



// ==========================
// ===== CPU СОРТИРОВКИ =====
// ==========================

// Последовательная быстрая сортировка на CPU (используем std::sort)
void cpuQuick(std::vector<int>& a) {
    // std::sort реализует эффективную гибридную quicksort / introsort
    std::sort(a.begin(), a.end());
}

// Последовательная пирамидальная сортировка на CPU
void cpuHeap(std::vector<int>& a) {
    // Превращаем массив в кучу
    std::make_heap(a.begin(), a.end());
    // Извлекаем элементы из кучи, формируя отсортированный массив
    std::sort_heap(a.begin(), a.end());
}



// ==========================
// ===== ИЗМЕРЕНИЕ ВРЕМЕНИ =====
// ==========================

// Шаблонная функция, измеряющая время выполнения переданной функции
template<class F>
double time_ms(F f) {
    // Фиксируем время начала выполнения
    auto t1 = clk::now();
    // Выполняем измеряемую функцию
    f();
    // Фиксируем время окончания выполнения
    auto t2 = clk::now();
    // Возвращаем разницу во времени в миллисекундах
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}



// ==========================
// ===== GPU ОБЁРТКИ =====
// ==========================

// Объявляем функцию GPU-версии быстрой сортировки
void gpuQuick(std::vector<int>& a);

// Объявляем функцию GPU-версии пирамидальной сортировки
void gpuHeap(std::vector<int>& a);



// ==========================
// ===== ТЕСТИРОВАНИЕ =====
// ==========================

// Функция тестирования для массива размера N
void test(int N) {
    // Создаём базовый массив на CPU
    std::vector<int> base(N);

    // Создаём генератор случайных чисел
    std::mt19937 rng(1);

    // Диапазон случайных значений
    std::uniform_int_distribution<int> d(0, 1000000);

    // Заполняем массив случайными числами
    for (int i = 0; i < N; ++i)
        base[i] = d(rng);

    // Копируем массив для CPU quicksort
    auto a = base;

    // Измеряем время CPU quicksort
    double tq = time_ms([&]() { cpuQuick(a); });

    // Восстанавливаем массив
    a = base;

    // Измеряем время CPU heapsort
    double th = time_ms([&]() { cpuHeap(a); });

    // Восстанавливаем массив
    a = base;

    // Измеряем время GPU quicksort
    double gq = time_ms([&]() { gpuQuick(a); });

    // Восстанавливаем массив
    a = base;

    // Измеряем время GPU heapsort
    double gh = time_ms([&]() { gpuHeap(a); });

    // Выводим размер теста
    std::cout << "\nN = " << N << "\n";

    // Выводим время CPU quicksort
    std::cout << "CPU Quick: " << tq << " ms\n";

    // Выводим время CPU heapsort
    std::cout << "CPU Heap : " << th << " ms\n";

    // Выводим время GPU quicksort
    std::cout << "GPU Quick: " << gq << " ms\n";

    // Выводим время GPU heapsort
    std::cout << "GPU Heap : " << gh << " ms\n";
}



// ==========================
// ===== MAIN =====
// ==========================

// Главная функция программы
int main() {

    // Тест для 10 тысяч элементов
    test(10000);

    // Тест для 100 тысяч элементов
    test(100000);

    // Тест для 1 миллиона элементов
    test(1000000);
}