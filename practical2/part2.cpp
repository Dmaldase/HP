#include <iostream>      // потоковый вывод результатов
#include <chrono>        // измерение времени выполнения сортировок
#include <cstdlib>       // rand(), srand()
#include <ctime>         // time()
#include <omp.h>         // OpenMP API: omp_set_num_threads, директивы parallel/for

//=====================================================================
//                ПОСЛЕДОВАТЕЛЬНЫЕ ВЕРСИИ СОРТИРОВОК
//=====================================================================

// Bubble Sort (O(N^2) сравнений и перестановок).
// Проходит массив, попарно меняя соседние элементы, крупные "всплывают".
void bubbleSort(int* arr, size_t n) {

    // внешний цикл — количество фаз, каждая уменьшает ненужный хвост
    for(size_t i = 0; i < n - 1; i++) {

        // внутренний цикл проходит соседние элементы
        // arr[j] и arr[j+1] читаются из кучи; swap записывает в кучу
        for(size_t j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j+1])
                std::swap(arr[j], arr[j+1]);  // обмен значений по адресам
        }
    }
}


// Selection Sort: при i-м шаге ищет минимальный элемент оставшейся части.
// Находит минимум и меняет arr[i] и arr[min_index].
void selectionSort(int* arr, size_t n) {
    for(size_t i = 0; i < n - 1; i++) {

        size_t min_index = i;   // индекс текущего минимума

        // поиск минимума на правой части массива
        for(size_t j = i + 1; j < n; j++) {
            if(arr[j] < arr[min_index])
                min_index = j;
        }

        std::swap(arr[i], arr[min_index]);
    }
}


// Insertion Sort: элементы слева поддерживаются упорядоченными.
// arr[i] вставляется на нужную позицию сдвигом элементов.
void insertionSort(int* arr, size_t n) {
    for(size_t i = 1; i < n; i++) {

        int key = arr[i];    // сохраняем значение
        int j = i - 1;

        // сдвиг элементов вправо, пока > key
        while(j >= 0 && arr[j] > key){
            arr[j+1] = arr[j];
            j--;
        }

        // вставляем key на найденную позицию
        arr[j+1] = key;
    }
}


//=====================================================================
//                        ПАРАЛЛЕЛЬНЫЕ ВЕРСИИ
//=====================================================================

// Корректный параллельный bubble через odd–even sort.
// На phase меняется старт сравнения: 0→парные пары, 1→непарные.
// Это устраняет гонки и фиксирует порядок обращений.
void bubbleSort_parallel(int* arr, size_t n){

    // N фаз гарантируют упорядочивание
    for(size_t phase=0; phase<n; phase++){

        // определяем старт сравнения
        // phase%2==0 → j={0,2,4,...}
        // phase%2==1 → j={1,3,5,...}
        size_t start = phase % 2;

        // параллельный цикл, j увеличивается на 2, соседи независимы
#pragma omp parallel for
for(size_t j = start; j < n - 1; j += 2){
    if(arr[j] > arr[j+1])
        std::swap(arr[j],arr[j+1]);
}
    }
}


// Parallel Selection Sort: на каждом шаге параллельно ищем минимум.
// Затем, после синхронизации, обмен arr[i] и глобального минимума.
// Гонки исключены через локальную копию и critical.
void selectionSort_parallel(int* arr, size_t n){

    for(size_t i=0;i<n-1;i++){

        // общий минимум в пределах этой итерации
        int global_min = arr[i];
        size_t global_idx = i;

        // Параллельная область: создаётся пул потоков
        #pragma omp parallel
        {
            // локальные копии min, чтобы избежать гонок чтения/записи
            int local_min = global_min;
            size_t local_idx = global_idx;

            // распределяем диапазон поиска минимума между потоками
            #pragma omp for nowait
            for(size_t j=i+1; j<n; j++){
                if(arr[j] < local_min){
                    local_min = arr[j];
                    local_idx = j;
                }
            }

            // критическая секция — поток-победитель обновляет глобальный минимум
            #pragma omp critical
            {
                if(local_min < global_min){
                    global_min = local_min;
                    global_idx = local_idx;
                }
            }
        }

        // обмен arr[i] и arr[global_idx] выполняется после параллельного блока,
        // когда есть корректный глобальный минимум
        std::swap(arr[i], arr[global_idx]);
    }
}


// Наивная параллельная версия вставками.
// НЕ обеспечивает корректности: несколько потоков будут писать в пересекающиеся
// участки массива, нарушая частичный порядок.
// Эта версия демонстрирует "как делать нельзя".
void insertionSort_parallel(int* arr, size_t n){

    // распределяем позиции i между потоками, но модификации пересекаются
    #pragma omp parallel for
    for(size_t i=1;i<n;i++){

        int key = arr[i];
        int j = i - 1;

        // возможны гонки: несколько потоков обращаются к arr
        while(j >= 0 && arr[j] > key){
            arr[j+1] = arr[j];
            j--;
        }

        arr[j+1] = key; // запись в общую память без синхронизации
    }
}


//=====================================================================
//                    Функция-бенчмарк сортировок
//=====================================================================

// seq и par — указатели на сортирующие функции
// name — имя метода
void benchmark(
    void (*seq)(int*,size_t),
    void (*par)(int*,size_t),
    size_t N,
    const char* name)
{
    // выделяем два массива под одну и ту же исходную выборку
    // arr seq и arr par должны сортировать исходно одинаковый массив
    int* a = new int[N];   // heap allocation
    int* b = new int[N];   // heap allocation

    // заполняем одинаковыми значениями
    for(size_t i=0;i<N;i++){
        int x = rand();   // rand() из stdlib, небезопасный, но приемлемый для теста
        a[i] = x;
        b[i] = x;
    }

    // измеряем последовательную версию
    auto t1 = std::chrono::high_resolution_clock::now();
    seq(a,N);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto seq_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    // измеряем параллельную версию
    auto t3 = std::chrono::high_resolution_clock::now();
    par(b,N);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto par_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();

    // выводим результаты
    // seq_time и par_time — wall-clock время
    std::cout<<name<<"   N="<<N
             <<"   seq="<<seq_time<<" ms"
             <<"   par="<<par_time<<" ms\n";

    // освобождаем массивы, уничтожаются в куче
    delete[] a;
    delete[] b;
}


int main(){

    // инициализация rand() значением на основе времени
    srand(time(NULL));

    // глобальная установка числа потоков OpenMP
    omp_set_num_threads(8);

    // тестируем три размера массива
    for(size_t N : {1000,10000,100000})
    {
        benchmark(bubbleSort, bubbleSort_parallel, N, "Bubble");
        benchmark(selectionSort, selectionSort_parallel, N, "Selection");
        benchmark(insertionSort, insertionSort_parallel, N, "Insertion");
        std::cout<<"-------------------------------------\n";
    }

    return 0; // стек очищен, память от heap освобождена внутри benchmark
}