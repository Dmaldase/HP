#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

int main() {
    const size_t N = 5'000'000;
    std::vector<int> arr(N);

    // заполнение массива случайными числами
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 100);
    for (size_t i = 0; i < N; ++i) {
        arr[i] = dist(gen);
    }

    // последовательное вычисление среднего
    auto t1 = std::chrono::high_resolution_clock::now();
    long long sum_seq = 0;
    for (size_t i = 0; i < N; ++i) {
        sum_seq += arr[i];
    }
    double avg_seq = static_cast<double>(sum_seq) / N;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_seq =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // параллельное вычисление среднего
    auto t3 = std::chrono::high_resolution_clock::now();
    long long sum_par = 0;

    #pragma omp parallel for reduction(+:sum_par)
    for (size_t i = 0; i < N; ++i) {
        sum_par += arr[i];
    }

    double avg_par = static_cast<double>(sum_par) / N;
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt_par =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "Sequential average = " << avg_seq << "\n";
    std::cout << "Parallel average   = " << avg_par << "\n";
    std::cout << "Sequential time = " << dt_seq << " us\n";
    std::cout << "Parallel time   = " << dt_par << " us\n";

    return 0;
}