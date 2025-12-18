#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

// пузырьковая сортировка
void bubbleSort(int* arr, size_t n) {
    for(size_t i = 0; i < n - 1; i++) {
        for(size_t j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j+1]) {
                std::swap(arr[j], arr[j+1]);
            }
        }
    }
}

// сортировка выбором
void selectionSort(int* arr, size_t n) {
    for(size_t i = 0; i < n - 1; i++) {
        size_t min_index = i;
        for(size_t j = i + 1; j < n; j++) {
            if(arr[j] < arr[min_index]) {
                min_index = j;
            }
        }
        std::swap(arr[i], arr[min_index]);
    }
}

// сортировка вставками
void insertionSort(int* arr, size_t n) {
    for(size_t i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while(j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}

void printArray(int* arr, size_t n) {
    for(size_t i = 0; i < n; i++){
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

int main() {

    std::srand(std::time(nullptr));

    const size_t N = 20;
    int arr[N];

    // заполнение случайными значениями
    for(size_t i = 0; i < N; i++){
        arr[i] = std::rand() % 100;
    }

    std::cout << "Исходный массив:\n";
    printArray(arr, N);

    // ----- пузырьком -----
    int arr_bub[N];
    std::copy(arr, arr + N, arr_bub);
    bubbleSort(arr_bub, N);
    std::cout << "\nПосле bubbleSort:\n";
    printArray(arr_bub, N);

    // ----- выбором -----
    int arr_sel[N];
    std::copy(arr, arr + N, arr_sel);
    selectionSort(arr_sel, N);
    std::cout << "\nПосле selectionSort:\n";
    printArray(arr_sel, N);

    // ----- вставками -----
    int arr_ins[N];
    std::copy(arr, arr + N, arr_ins);
    insertionSort(arr_ins, N);
    std::cout << "\nПосле insertionSort:\n";
    printArray(arr_ins, N);

    return 0;
}
