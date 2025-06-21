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

void print_dataset(std::vector<float> inputs, std::vector<float> labels, int input_dim, int num_samples)
{
    for (int i = 0; i < num_samples; i++)
    {
        std::cout << "Label: " << labels[i] << ". Pixels: ";

        int idx = i * input_dim;

        for (int j = 0; j < 10; j++)
        {
            float pixel = inputs[idx + j];
            std::cout << pixel << ", ";
        }
        std::cout << "..., " << inputs[idx + input_dim - 1] << "\n";
    }
    std::cout << std::endl;
}
