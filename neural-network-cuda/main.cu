
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "linear_layer.h"

void print_matrix(const float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    int batch_size = 2;
    int input_dim = 4;
    int output_dim = 3;

    // Allocate and initialize input on host
    float h_input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    // Allocate device memory for input
    float* d_input;
    cudaMalloc(&d_input, sizeof(float) * batch_size * input_dim);
    cudaMemcpy(d_input, h_input, sizeof(float) * batch_size * input_dim, cudaMemcpyHostToDevice);

    // Create and run linear layer
    LinearLayer layer(input_dim, output_dim);
    layer.forward(d_input, batch_size);

    // Copy output back to host
    float* h_output = new float[batch_size * output_dim];
    cudaMemcpy(h_output, layer.d_output, sizeof(float) * batch_size * output_dim, cudaMemcpyDeviceToHost);

    std::cout << "Forward output:\n";
    print_matrix(h_output, batch_size, output_dim);

    // Cleanup
    cudaFree(d_input);
    delete[] h_output;
}