#include <iostream>   // вывод результатов
#include <chrono>     // измерение времени
#include <omp.h>      // OpenMP

// Узел односвязной структуры
struct Node {
    int data;         // значение
    Node* next;       // указатель на следующий узел
};

// Односвязный список: вставка в голову
struct SinglyList {
    Node* head;

    SinglyList() : head(nullptr) {}

    // push_front: создаёт новый узел и делает его головным
    // выделение памяти выполняется в куче (new)
    void push_front(int value) {
        Node* node = new Node{value, head};
        head = node;
    }
};

// Стек: push в вершину (односвязный список)
struct Stack {
    Node* top = nullptr;

    void push(int value) {
        Node* node = new Node{value, top};
        top = node;
    }
};

// Очередь: enqueue в конец
struct Queue {
    Node* front = nullptr;
    Node* back  = nullptr;

    // добавление узла в конец
    void enqueue(int value) {
        Node* node = new Node{value, nullptr};
        if (back)
            back->next = node;   // связать конец со следующим
        else
            front = node;        // очередь была пустой
        back = node;             // обновить конец
    }
};

int main() {

    const int N = 1'000'000;         // число операций вставки

    int thread_list[4] = {1,2,4,8};  // числа потоков для теста

    // цикл по конфигурациям числа потоков
    for (int idx = 0; idx < 4; ++idx) {

        int threads = thread_list[idx];
        omp_set_num_threads(threads);     // установка числа потоков OpenMP

        // создаются три пустые структуры
        SinglyList lst;
        Queue q;
        Stack st;

        // измерение enqueue в очередь
        auto t1 = std::chrono::high_resolution_clock::now();

        // параллельный цикл: несколько потоков выполняют тело
        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            // критическая секция — только один поток в момент времени
            // защищает конкурентный доступ к структуре
            #pragma omp critical
            q.enqueue(i);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        long long dt_q =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        // измерение push_front в список
        auto t3 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            #pragma omp critical
            lst.push_front(i);        // аллокация узла и вставка в голову
        }

        auto t4 = std::chrono::high_resolution_clock::now();
        long long dt_lst =
            std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

        // измерение push стека
        auto t5 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            #pragma omp critical
            st.push(i);               // аллокация и вставка в вершину
        }

        auto t6 = std::chrono::high_resolution_clock::now();
        long long dt_st =
            std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

        // вывод замеров времени
        std::cout << "\nthreads = " << threads << "\n";
        std::cout << "queue enqueue time = " << dt_q  << " ms\n";
        std::cout << "list  push time    = " << dt_lst << " ms\n";
        std::cout << "stack push time    = " << dt_st << " ms\n";
        std::cout << "--------------------------------------\n";
    }

    return 0;
}