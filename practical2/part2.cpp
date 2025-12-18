#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// ================= SEQUENTIAL SORTS =====================

// bubble
void bubbleSort(int* arr, size_t n) {
    for(size_t i = 0; i < n - 1; i++) {
        for(size_t j = 0; j < n - i - 1; j++) {
            if(arr[j] > arr[j+1])
                std::swap(arr[j], arr[j+1]);
        }
    }
}

// selection
void selectionSort(int* arr, size_t n) {
    for(size_t i = 0; i < n - 1; i++) {
        size_t min_index = i;
        for(size_t j = i + 1; j < n; j++) {
            if(arr[j] < arr[min_index])
                min_index = j;
        }
        std::swap(arr[i], arr[min_index]);
    }
}

// insertion
void insertionSort(int* arr, size_t n) {
    for(size_t i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while(j >= 0 && arr[j] > key){
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}


// ================= PARALLEL SORTS =======================

// корректный параллельный bubble через odd-even
void bubbleSort_parallel(int* arr, size_t n){
    for(size_t phase=0; phase<n; phase++){
        size_t start = phase % 2;
        #pragma omp parallel for
        for(size_t j=start; j+1<n; j+=2){
            if(arr[j] > arr[j+1])
                std::swap(arr[j],arr[j+1]);
        }
    }
}

// корректный selection sort: параллельный поиск минимума
void selectionSort_parallel(int* arr, size_t n){
    for(size_t i=0;i<n-1;i++){

        int global_min = arr[i];
        size_t global_idx = i;

        #pragma omp parallel
        {
            int local_min = global_min;
            size_t local_idx = global_idx;

            #pragma omp for nowait
            for(size_t j=i+1; j<n; j++){
                if(arr[j] < local_min){
                    local_min = arr[j];
                    local_idx = j;
                }
            }

            #pragma omp critical
            {
                if(local_min < global_min){
                    global_min = local_min;
                    global_idx = local_idx;
                }
            }
        }

        std::swap(arr[i], arr[global_idx]);
    }
}

// наивная параллельная insertion (НЕ гарантирует корректность)
void insertionSort_parallel(int* arr, size_t n){
    #pragma omp parallel for
    for(size_t i=1;i<n;i++){
        int key = arr[i];
        int j = i - 1;
        while(j >= 0 && arr[j] > key){
            arr[j+1] = arr[j];
            j--;
        }
        arr[j+1] = key;
    }
}


// ================= BENCHMARK =============================

void benchmark(
    void (*seq)(int*,size_t),
    void (*par)(int*,size_t),
    size_t N,
    const char* name)
{
    int* a = new int[N];
    int* b = new int[N];

    for(size_t i=0;i<N;i++){
        int x = rand();
        a[i] = x;
        b[i] = x;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    seq(a,N);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto seq_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    auto t3 = std::chrono::high_resolution_clock::now();
    par(b,N);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto par_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();

    std::cout<<name<<"   N="<<N
             <<"   seq="<<seq_time<<" ms"
             <<"   par="<<par_time<<" ms\n";

    delete[] a;
    delete[] b;
}


int main(){
    srand(time(NULL));
    omp_set_num_threads(8);

    for(size_t N : {1000,10000,100000})
    {
        benchmark(bubbleSort, bubbleSort_parallel, N, "Bubble");
        benchmark(selectionSort, selectionSort_parallel, N, "Selection");
        benchmark(insertionSort, insertionSort_parallel, N, "Insertion");
        std::cout<<"-------------------------------------\n";
    }

    return 0;
}
