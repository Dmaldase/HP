#include <iostream>      // потоковый ввод/вывод: std::cin/std::cout
#include <vector>        // std::vector — динамический массив (heap allocation)
#include <random>        // генератор случайных чисел std::mt19937
#include <chrono>        // std::chrono::high_resolution_clock для замеров времени
#include <omp.h>         // заголовок OpenMP и директивы параллельных циклов

int main() {

    size_t N;                                      // локальная переменная в стековом кадре main
    std::cout << "Input N: ";                   // запись в stdout (буферизирован)
    std::cin >> N;                                // чтение числа, возможен блокирующий ввод

    // создаётся объект std::vector в стеке
    // внутри в куче (heap) выделяется блок памяти размером N*sizeof(int)
    std::vector<int> arr(N);

    // генератор псевдослучайных чисел Мерсенна
    // хранит состояние PRNG (~2.5 KB) внутри объекта на стеке
    std::mt19937 gen(std::random_device{}());

    // равномерное распределение целых чисел от 1 до 100
    std::uniform_int_distribution<int> dist(1,100);

    // последовательное заполнение массива
    // arr[i] — обращение к элементу массива в куче
    for(size_t i=0;i<N;i++){
        arr[i] = dist(gen);                       // каждый вызов dist генерирует число
    }

    // при малом N выводим элементы, чтобы убедиться в корректности
    if(N <= 1000) {
        std::cout << "Array: ";
        for(size_t i=0;i<N;i++){
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

    // фиксируем момент времени перед последовательным поиском min/max
    auto t1 = std::chrono::high_resolution_clock::now();

    // инициализация min/max значениями первого элемента массива
    int mn_seq = arr[0];
    int mx_seq = arr[0];

    // последовательный проход массива
    // чтение arr[i] из heap и сравнение со значениями в регистрах
    for(size_t i=1;i<N;i++){
        if(arr[i] < mn_seq) mn_seq = arr[i];
        if(arr[i] > mx_seq) mx_seq = arr[i];
    }

    // фиксируем момент времени после завершения последовательного цикла
    auto t2 = std::chrono::high_resolution_clock::now();

    // разница двух time_point даёт duration
    // duration_cast<...> приводит её к микросекундам
    auto dt_seq =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // вывод последовательных результатов
    std::cout << "Seq min = " << mn_seq << "\n";
    std::cout << "Seq max = " << mx_seq << "\n";
    std::cout << "Seq time = " << dt_seq << " us\n";

    // фиксируем момент времени перед параллельным поиском
    auto t3 = std::chrono::high_resolution_clock::now();

    // инициализация глобальных min/max для reduction
    // переменные размещены в стеке вызывающего потока
    int mn_par = arr[0];
    int mx_par = arr[0];

    // parallel for — создаёт пул потоков
    // каждая нить получает поддиапазон индексов i
    // reduction(min:mn_par):
    //   создаёт приватные копии mn_local в каждом потоке
    //   инициализация mn_local значением mn_par
    //   в конце — выберется минимальное по всем mn_local
    // reduction(max:mx_par):
    //   аналогично для mx_par
    #pragma omp parallel for reduction(min:mn_par) reduction(max:mx_par)
    for(size_t i=1;i<N;i++){

        // чтение arr[i] безопасно: множество потоков читают — гонок нет
        // запись в mn_par/mx_par фактически идёт в приватные копии
        if(arr[i] < mn_par) mn_par = arr[i];
        if(arr[i] > mx_par) mx_par = arr[i];
    }

    // после завершения parallel-for происходит фаза reduce:
    //   min объединяет минимумы всех потоков
    //   max объединяет максимумы всех потоков
    // результаты пишутся в mn_par и mx_par в главном потоке

    auto t4 = std::chrono::high_resolution_clock::now();

    auto dt_par =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    // вывод параллельных результатов
    std::cout << "Parallel min = " << mn_par << "\n";
    std::cout << "Parallel max = " << mx_par << "\n";
    std::cout << "Parallel time = " << dt_par << " us\n";

    return 0;  
    // стековый кадр main уничтожается
    // объект vector вызывает деструктор, освобождая память из heap
}