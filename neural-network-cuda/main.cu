#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cassert>

#include "layers/linear_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"

void print_matrix(const float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

void test_linear_backward() {
    const int batch_size = 3;
    const int input_dim = 3;
    const int output_dim = 3;
    const float lr = 0.1f;

    // Create layer
    Linear layer(input_dim, output_dim);

    // Allocate and fill dummy input
    float h_input[batch_size * input_dim] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float* d_input;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float) * batch_size * input_dim));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float) * batch_size * input_dim, cudaMemcpyHostToDevice));

    // Forward pass
    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float) * batch_size * output_dim));
    layer.forward(d_input, batch_size);

    // Allocate dummy output gradient (e.g., from MSE loss w.r.t. output)
    float h_dY[batch_size * output_dim] = {
        0.1f, -0.2f,
        0.05f, 0.3f
    };
    float* d_dY;
    CHECK_CUDA(cudaMalloc(&d_dY, sizeof(float) * batch_size * output_dim));
    CHECK_CUDA(cudaMemcpy(d_dY, h_dY, sizeof(float) * batch_size * output_dim, cudaMemcpyHostToDevice));

    // Input gradient output
    float* d_dX;
    CHECK_CUDA(cudaMalloc(&d_dX, sizeof(float) * batch_size * input_dim));

    // Store original weights and biases
    float h_W_before[input_dim * output_dim], h_b_before[output_dim];
    CHECK_CUDA(cudaMemcpy(h_W_before, layer.d_weights, sizeof(float) * input_dim * output_dim, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_before, layer.d_bias, sizeof(float) * output_dim, cudaMemcpyDeviceToHost));

    // Backward pass
    layer.backward(d_dY, d_dX, lr, batch_size);

    // Get updated weights and biases
    float h_W_after[input_dim * output_dim], h_b_after[output_dim];
    CHECK_CUDA(cudaMemcpy(h_W_after, layer.d_weights, sizeof(float) * input_dim * output_dim, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_after, layer.d_bias, sizeof(float) * output_dim, cudaMemcpyDeviceToHost));

    // Print diffs
    std::cout << "Weight updates:\n";
    for (int i = 0; i < input_dim * output_dim; ++i)
        std::cout << "W[" << i << "]: " << h_W_before[i] << " -> " << h_W_after[i] << "\n";

    std::cout << "Bias updates:\n";
    for (int i = 0; i < output_dim; ++i)
        std::cout << "b[" << i << "]: " << h_b_before[i] << " -> " << h_b_after[i] << "\n";

    // Check input gradient is not all zero
    float h_dX[batch_size * input_dim];
    CHECK_CUDA(cudaMemcpy(h_dX, d_dX, sizeof(h_dX), cudaMemcpyDeviceToHost));

    bool non_zero = false;
    for (int i = 0; i < batch_size * input_dim; ++i) {
        if (std::abs(h_dX[i]) > 1e-6) {
            non_zero = true;
            break;
        }
    }

    assert(non_zero && "Input gradient is all zeros — backward may have failed!");

    std::cout << "Backward pass test passed.\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dY);
    cudaFree(d_dX);
}

void test_relu_backward() 
{
    const int batch_size = 2;
    const int dim = 4;
    const int size = batch_size * dim;

    // Host input
    float h_input[size] = { -1.0f, 2.0f, -3.0f, 4.0f,
                             5.0f, -6.0f, 7.0f, -8.0f };

    float h_grad_out[size] = { 1.0f, 1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f, 1.0f };

    float h_output[size], h_grad_input[size];

    // Device memory
    float* d_input, * d_grad_out, * d_grad_input;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_grad_out, size * sizeof(float));
    cudaMalloc(&d_grad_input, size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_out, h_grad_out, size * sizeof(float), cudaMemcpyHostToDevice);

    // Create and run ReLU
    ReLU relu(dim);
    relu.forward(d_input, batch_size);
    float* d_output = relu.get_output();

    // Copy result back
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "ReLU forward output:\n";
    print_matrix(h_output, 1, size);

    // Run backward
    relu.backward(d_grad_out, d_grad_input, 0.0f, batch_size);
    cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "ReLU backward grad_input:\n";
    print_matrix(h_grad_input, 1, size);

    // Free
    cudaFree(d_input);
    cudaFree(d_grad_out);
    cudaFree(d_grad_input);
}

void test_sigmoid_backward()
{
    const int batch_size = 2;
    const int dim = 4;
    const int size = batch_size * dim;

    float h_input[size] = {
        -2.0f, 0.0f, 1.0f, 2.0f,
         3.0f, -1.0f, -4.0f, 0.5f
    };

    float h_output_grad[size] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };

    // Allocate device memory
    float* d_input, * d_output_grad, * d_input_grad;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output_grad, size * sizeof(float));
    cudaMalloc(&d_input_grad, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_grad, h_output_grad, size * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize sigmoid layer
    Sigmoid sigmoid(dim);

    // Forward
    sigmoid.forward(d_input, batch_size);
    float* d_output = sigmoid.get_output();
    float h_output[size];
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sigmoid Forward Output: \n";
    print_matrix(h_output, batch_size, dim);

    // Backward
    sigmoid.backward(d_output_grad, d_input_grad, 0.0f, batch_size);
    float h_input_grad[size];
    cudaMemcpy(h_input_grad, d_input_grad, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sigmoid Backward d_input_grad: \n";
    print_matrix(h_input_grad, batch_size, dim);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output_grad);
    cudaFree(d_input_grad);
}

int main()
{
    test_linear_backward();
    test_relu_backward();
    test_sigmoid_backward();

    //TODO: Implement Binary Cross Entropy Loss

    return 0;
}