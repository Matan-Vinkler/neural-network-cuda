#include "utils.h"

#include <cuda_runtime.h>
#include <iostream>

void print_h_matrix(const float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void print_d_matrix(const float* d_data, int rows, int cols)
{
    float* h_data = new float[rows * cols];
    cudaMemcpy(h_data, d_data, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);

    print_h_matrix(h_data, rows, cols);

    delete[] h_data;
}