#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    size_t N;
    std::cout << "Введите N: ";
    std::cin >> N;

    std::vector<int> arr(N);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1,100);

    for(size_t i=0;i<N;i++){
        arr[i] = dist(gen);
    }

    if(N <= 1000) {
        std::cout << "Массив: ";
        for(size_t i=0;i<N;i++){
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int mn_seq = arr[0];
    int mx_seq = arr[0];

    for(size_t i=1;i<N;i++){
        if(arr[i] < mn_seq) mn_seq = arr[i];
        if(arr[i] > mx_seq) mx_seq = arr[i];
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    auto dt_seq =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Последовательный min = " << mn_seq << "\n";
    std::cout << "Последовательный max = " << mx_seq << "\n";
    std::cout << "Время последовательной версии = " << dt_seq << " us\n";

    auto t3 = std::chrono::high_resolution_clock::now();

    int mn_par = arr[0];
    int mx_par = arr[0];

    #pragma omp parallel for reduction(min:mn_par) reduction(max:mx_par)
    for(size_t i=1;i<N;i++){
        if(arr[i] < mn_par) mn_par = arr[i];
        if(arr[i] > mx_par) mx_par = arr[i];
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    auto dt_par =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "Параллельный min = " << mn_par << "\n";
    std::cout << "Параллельный max = " << mx_par << "\n";
    std::cout << "Время параллельной версии = " << dt_par << " us\n";

    return 0;
}