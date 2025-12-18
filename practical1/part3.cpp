#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

double sequential_avg(int* arr, size_t N) {
    long long sum = 0;
    for (size_t i = 0; i < N; i++) {
        sum += arr[i];
    }
    return static_cast<double>(sum) / N;
}

double parallel_avg(int* arr, size_t N) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < (long long)N; i++) {
        sum += arr[i];
    }
    return static_cast<double>(sum) / N;
}

int main() {

    size_t N = 5'000'000;

    // динамическое выделение памяти под массив
    int* arr = new int[N];

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(1,100);

    for(size_t i=0;i<N;i++){
        arr[i] = dist(gen);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_seq = sequential_avg(arr,N);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto seq_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    auto t3 = std::chrono::high_resolution_clock::now();
    double avg_par = parallel_avg(arr,N);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto par_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();

    std::cout << "Sequential average = " << avg_seq << "\n";
    std::cout << "Parallel average   = " << avg_par << "\n";
    std::cout << "Seq time = " << seq_time << " ms\n";
    std::cout << "Par time = " << par_time << " ms\n";

    delete[] arr;

    return 0;
}