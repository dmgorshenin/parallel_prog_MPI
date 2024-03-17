#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <time.h>
#include <omp.h>
#include "mpi.h"

using namespace std;
using namespace std::chrono;

class MultiplyMatrix {
private:
    int size = 500;
    int** matrix1;
    int** matrix2;
    int** result_matrix;
    string filename_matrix1 = "matrix1.txt";
    string filename_matrix2 = "matrix2.txt";
    string filename_matrix_res = "result_matrix.txt";

    int** create_matrix() {
        int** matrix = new int* [size];
        for (int i = 0; i < size; ++i)
            matrix[i] = new int[size];
        return matrix;
    }

    void generateRandomMatrix(int*** matrix) {
        srand(time(NULL));
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                (*matrix)[i][j] = rand() % 100;
            }
        }
    }

    void writeMatrixToFile(int** matrix, const string& filename, int rank) {
        if (rank == 0) {
            ofstream file(filename);
            if (!file.is_open()) {
                cerr << "Unable to open file: " << filename << endl;
                exit(1);
            }

            file << size << endl;
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    file << matrix[i][j] << " ";
                }
                file << endl;
            }

            file.close();
        }
    }

public:
    MultiplyMatrix(int N) {
        this->size = N;
        matrix1 = create_matrix();
        matrix2 = create_matrix();
        result_matrix = create_matrix();
    }

    ~MultiplyMatrix() {
        clean(&matrix1);
        clean(&matrix2);
        clean(&result_matrix);
    }

    void generate_and_save_matrices(int rank) {
        generateRandomMatrix(&matrix1);
        generateRandomMatrix(&matrix2);

        if (rank == 0) {
            writeMatrixToFile(matrix1, filename_matrix1, rank);
            writeMatrixToFile(matrix2, filename_matrix2, rank);
        }
    }

    void multiplyMatrices(int rank, int num_procs) {
        int local_size = size / num_procs;

        int* matrix2_1d = new int[size * size];
        int* matrix1_1d = new int[size * size];
        int* matrix_result_1d = new int[size * size];

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix1_1d[i * size + j] = matrix1[i][j];
                matrix2_1d[i * size + j] = matrix2[i][j];
                matrix_result_1d[i * size + j] = 0;
            }
        }

        int* local_matrix2 = new int[local_size * size];
        int* local_matrix1 = new int[local_size * size];
        int* local_result_matrix = new int[local_size * size];

        MPI_Bcast(matrix2_1d, size * size, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Scatter(matrix1_1d, local_size * size, MPI_INT, local_matrix1, local_size * size, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < local_size; ++i) {
            for (int j = 0; j < size; ++j) {
                local_result_matrix[i * size + j] = 0;
                for (int k = 0; k < size; ++k) {
                    local_result_matrix[i * size + j] += local_matrix1[i * size + k] * matrix2_1d[k * size + j];
                }
            }
        }

        MPI_Gather(local_result_matrix, local_size * size, MPI_INT, matrix_result_1d, local_size * size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    result_matrix[i][j] = matrix_result_1d[i * size + j];
                }
            }
            writeMatrixToFile(result_matrix, filename_matrix_res, rank);
        }

        delete[] local_matrix1;
        delete[] local_matrix2;
        delete[] local_result_matrix;
        delete[] matrix2_1d;
        delete[] matrix1_1d;
        delete[] matrix_result_1d;
    }


    void clean(int*** matrix) {
        for (int i = 0; i < size; ++i) {
            delete[](*matrix)[i];
        }
        delete[](*matrix);
    }
};

void writeTimeToFile(long long computation_time, const string& filename, int rank) {
    if (rank == 0) {
        ofstream file(filename, ios::app);
        if (!file.is_open()) {
            cerr << "Unable to open file: " << filename << endl;
            exit(1);
        }

        file << computation_time << endl;

        file.close();
    }
}

void writeTaskSizeToFile(int size, long long task_size, const string& filename, int rank) {
    if (rank == 0) {
        ofstream file(filename, ios::app);
        if (!file.is_open()) {
            cerr << "Unable to open file: " << filename << endl;
            exit(1);
        }

        file << size << endl << task_size << endl;

        file.close();
    }
}


int main(int argc, char** argv) {
    setlocale(LC_ALL, "ru");
    int N = 500;
    string file_stat = "stats_mpi8.txt";

    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    while (N <= 2300) {
        if (rank == 0) {
            long long task_size = static_cast<long long>(N) * static_cast<long long>(N) * static_cast<long long>(N);
            writeTaskSizeToFile(N, task_size, file_stat, rank);

            cout << "Размер матриц " << N << "x" << N << endl << "Объем задачи: " << task_size << endl;
        }

        for (size_t i = 0; i < 10; ++i) {
            MultiplyMatrix matrix(N);
            if (rank == 0) {
                matrix.generate_and_save_matrices(rank);


                cout << "Файлы с данными для матриц сгенерированы" << endl;
            }

            auto start_compute = high_resolution_clock::now();
            matrix.multiplyMatrices(rank, num_procs);
            auto stop_compute = high_resolution_clock::now();

          if (rank == 0) {
                cout << "Матрицы перемножены." << endl;

                auto duration_computation = duration_cast<milliseconds>(stop_compute - start_compute);

                cout << "Время умножения матриц: " << duration_computation.count() << " мс" << endl;
                writeTimeToFile(duration_computation.count(), file_stat, rank);
            }
        }
        N += 100;
    }

    MPI_Finalize();
    return 0;
}