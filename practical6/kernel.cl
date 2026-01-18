__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int N, int M, int K)
{
    // Индексы элемента C
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Проверка границ
    if (row < N && col < K) {
        float sum = 0.0f;

        // Скалярное произведение строки A и столбца B
        for (int i = 0; i < M; i++) {
            sum += A[row * M + i] * B[i * K + col];
        }

        // Запись результата
        C[row * K + col] = sum;
    }
}
