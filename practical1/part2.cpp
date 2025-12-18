#include <iostream>
#include <chrono>
#include <omp.h>

struct Node {
    int data;
    Node* next;
};

// односвязный список
struct SinglyList {
    Node* head;

    SinglyList() : head(nullptr) {}

    void push_front(int value) {
        Node* node = new Node{value, head};
        head = node;
    }
};

// стек
struct Stack {
    Node* top = nullptr;

    void push(int value) {
        Node* node = new Node{value, top};
        top = node;
    }
};

// очередь
struct Queue {
    Node* front = nullptr;
    Node* back  = nullptr;

    void enqueue(int value) {
        Node* node = new Node{value, nullptr};
        if (back)
            back->next = node;
        else
            front = node;
        back = node;
    }
};

int main() {

    const int N = 1'000'000;

    int thread_list[4] = {1,2,4,8};

    for (int idx = 0; idx < 4; ++idx) {

        int threads = thread_list[idx];
        omp_set_num_threads(threads);

        SinglyList lst;
        Queue q;
        Stack st;

        auto t1 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            #pragma omp critical
            q.enqueue(i);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        long long dt_q =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        auto t3 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            #pragma omp critical
            lst.push_front(i);
        }

        auto t4 = std::chrono::high_resolution_clock::now();
        long long dt_lst =
            std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

        auto t5 = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0;i<N;i++) {
            #pragma omp critical
            st.push(i);
        }

        auto t6 = std::chrono::high_resolution_clock::now();
        long long dt_st =
            std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

        std::cout << "\nthreads = " << threads << "\n";
        std::cout << "queue enqueue time = " << dt_q  << " ms\n";
        std::cout << "list  push time    = " << dt_lst << " ms\n";
        std::cout << "stack push time    = " << dt_st << " ms\n";
        std::cout << "--------------------------------------\n";
    }

    return 0;
}
