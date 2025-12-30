#include <iostream>   // потоковый ввод/вывод
#include <vector>     // динамический массив
#include <random>     // генератор случайных чисел
#include <chrono>     // точное измерение времени
#include <omp.h>      // OpenMP: параллельные директивы

int main() {
    size_t N;                                     // размер массива, хранится в стеке
    std::cout << "Input N: ";
    std::cin >> N;                                // чтение числа из stdin

    std::vector<int> arr(N);                      // создаётся объект vector
                                                  // vector запрашивает из кучи N*sizeof(int) байт
                                                  // указатель, размер и capacity хранятся внутри arr

    std::mt19937 gen(std::random_device{}());     // генератор Мерсенна, состояние внутри объекта
    std::uniform_int_distribution<int> dist(1,100); // равномерное распределение целых [1;100]

    // последовательное заполнение массива
    for(size_t i=0;i<N;i++){
        arr[i] = dist(gen);                       // генерация и запись значения в кучу
    }

    // если массив небольшой, выводим его элементы
    if(N <= 1000) {
        std::cout << "Array: ";
        for(size_t i=0;i<N;i++){
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

    // измеряем время последовательного прохода
    auto t1 = std::chrono::high_resolution_clock::now();

    int mn_seq = arr[0];                           // текущий минимум
    int mx_seq = arr[0];                           // текущий максимум

    // последовательное вычисление min/max
    for(size_t i=1;i<N;i++){
        if(arr[i] < mn_seq) mn_seq = arr[i];
        if(arr[i] > mx_seq) mx_seq = arr[i];
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // вычисление времени: t2 - t1 (микросекунды)
    auto dt_seq =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Seq min = " << mn_seq << "\n";
    std::cout << "Seq max = " << mx_seq << "\n";
    std::cout << "Seq time = " << dt_seq << " us\n";

    // измеряем время параллельного прохода
    auto t3 = std::chrono::high_resolution_clock::now();

    int mn_par = arr[0];
    int mx_par = arr[0];

    // параллельный цикл OpenMP с редукциями по min/max
    // создаются локальные копии переменных mn_par/mx_par в каждом потоке
    // по завершении цикла выполняется редукция: объединение локальных min/max
    #pragma omp parallel for reduction(min:mn_par) reduction(max:mx_par)
    for(size_t i=1;i<N;i++){
        if(arr[i] < mn_par) mn_par = arr[i];      // запись только в локальное значение
        if(arr[i] > mx_par) mx_par = arr[i];
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    auto dt_par =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "Parallel min = " << mn_par << "\n";
    std::cout << "Parallel max = " << mx_par << "\n";
    std::cout << "Parallel time = " << dt_par << " us\n";

    return 0;  // стек очищается, vector вызывает деструктор, освобождая память из кучи
}