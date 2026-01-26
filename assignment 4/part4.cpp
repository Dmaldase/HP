// Подключаем стандартную библиотеку ввода-вывода
#include <cstdio>

// Подключаем стандартную библиотеку для работы с памятью
#include <cstdlib>

// Подключаем MPI библиотеку
#include <mpi.h>

int main(int argc, char** argv)
{
    // Инициализация MPI окружения
    MPI_Init(&argc, &argv);

    // Получаем общее количество процессов
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Размер всего массива
    const int N = 1'000'000;

    // Размер локального куска для каждого процесса
    int localSize = N / worldSize;

    // Указатели на глобальные массивы (только у rank 0)
    float* globalInput  = nullptr;
    float* globalOutput = nullptr;

    // Выделяем глобальные массивы только в главном процессе
    if (rank == 0)
    {
        globalInput  = (float*)malloc(N * sizeof(float));
        globalOutput = (float*)malloc(N * sizeof(float));

        // Инициализируем входной массив
        for (int i = 0; i < N; i++)
            globalInput[i] = 1.0f;
    }

    // Выделяем локальные массивы для каждого процесса
    float* localInput  = (float*)malloc(localSize * sizeof(float));
    float* localOutput = (float*)malloc(localSize * sizeof(float));

    // Синхронизация всех процессов перед началом замеров
    MPI_Barrier(MPI_COMM_WORLD);

    // Засекаем время начала вычислений
    double startTime = MPI_Wtime();

    // Распределяем части массива между процессами
    MPI_Scatter(
        globalInput,          // исходный массив (rank 0)
        localSize,            // сколько элементов отправляется каждому
        MPI_FLOAT,            // тип данных
        localInput,           // локальный буфер
        localSize,            // размер локального буфера
        MPI_FLOAT,            // тип данных
        0,                    // корневой процесс
        MPI_COMM_WORLD        // коммуникатор
    );

    // Локальная обработка массива
    for (int i = 0; i < localSize; i++)
    {
        // Простая операция обработки
        localOutput[i] = localInput[i] * 2.0f;
    }

    // Сбор обработанных данных обратно в главный процесс
    MPI_Gather(
        localOutput,          // локальный массив
        localSize,            // сколько элементов отправляем
        MPI_FLOAT,            // тип данных
        globalOutput,         // результирующий массив (rank 0)
        localSize,            // размер куска от каждого процесса
        MPI_FLOAT,            // тип данных
        0,                    // корневой процесс
        MPI_COMM_WORLD        // коммуникатор
    );

    // Синхронизация перед окончанием замера
    MPI_Barrier(MPI_COMM_WORLD);

    // Засекаем время окончания
    double endTime = MPI_Wtime();

    // Вывод времени выполнения (только из главного процесса)
    if (rank == 0)
    {
        printf("Processes: %d\n", worldSize);
        printf("Execution time: %.6f seconds\n", endTime - startTime);
    }

    // Освобождаем локальную память
    free(localInput);
    free(localOutput);

    // Освобождаем глобальную память в главном процессе
    if (rank == 0)
    {
        free(globalInput);
        free(globalOutput);
    }

    // Завершаем MPI окружение
    MPI_Finalize();

    return 0;
}
