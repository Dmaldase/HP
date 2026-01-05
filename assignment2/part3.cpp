#include <iostream>              // стандартный ввод/вывод
#include <vector>                // контейнер для массива
#include <random>                // генерация случайных чисел
#include <chrono>                // измерение времени
#include <omp.h>                 // библиотека OpenMP

// Функция заполнения массива случайными числами
void fillArray(std::vector<int>& a) {
    std::mt19937 gen(std::random_device{}());           // генератор случайных чисел
    std::uniform_int_distribution<> dist(1, 100000);   // диапазон значений

    for (int& x : a)                                     // проходим по всему массиву
        x = dist(gen);                                   // присваиваем случайное число
}

// Последовательная сортировка выбором
void selectionSortSequential(std::vector<int>& a) {
    int n = a.size();                                   // размер массива

    for (int i = 0; i < n - 1; ++i) {                   // внешний цикл сортировки
        int minIndex = i;                               // считаем текущий элемент минимальным

        for (int j = i + 1; j < n; ++j) {               // ищем минимальный элемент справа
            if (a[j] < a[minIndex])
                minIndex = j;                           // обновляем индекс минимума
        }

        std::swap(a[i], a[minIndex]);                   // меняем местами элементы
    }
}

// Параллельная версия сортировки выбором
void selectionSortParallel(std::vector<int>& a) {
    int n = a.size();                                   // размер массива

    for (int i = 0; i < n - 1; ++i) {                   // внешний цикл остаётся последовательным
        int minIndex = i;                               // глобальный минимум

        #pragma omp parallel                            // создаём группу потоков
        {
            int localMinIndex = minIndex;               // локальный минимум для каждого потока

            #pragma omp for nowait                      // делим внутренний цикл между потоками
            for (int j = i + 1; j < n; ++j) {           // каждый поток ищет свой локальный минимум
                if (a[j] < a[localMinIndex])
                    localMinIndex = j;
            }

            #pragma omp critical                        // синхронизация потоков
            {                                           // объединяем локальные минимумы
                if (a[localMinIndex] < a[minIndex])
                    minIndex = localMinIndex;
            }
        }

        std::swap(a[i], a[minIndex]);                   // фиксируем найденный минимум
    }
}

// Функция тестирования времени работы
double testSort(int n, bool parallel) {
    std::vector<int> arr(n);                            // создаём массив
    fillArray(arr);                                     // заполняем случайными числами

    auto start = std::chrono::high_resolution_clock::now(); // начало замера времени

    if (parallel)
        selectionSortParallel(arr);                     // параллельная версия
    else
        selectionSortSequential(arr);                   // последовательная версия

    auto end = std::chrono::high_resolution_clock::now();   // конец замера времени

    return std::chrono::duration<double>(end - start).count(); // возвращаем время работы
}

int main() {
    int sizes[2] = {1000, 10000};                       // размеры тестовых массивов

    for (int n : sizes) {                               // тестируем для каждого размера
        double t1 = testSort(n, false);                // последовательное выполнение
        double t2 = testSort(n, true);                 // параллельное выполнение

        std::cout << "Array size: " << n << "\n";
        std::cout << "Seq: " << t1 << " sec\n";
        std::cout << "Parallel:     " << t2 << " sec\n\n";
    }
}