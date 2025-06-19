#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cassert>

#include "layers/linear_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/sequential.h"
#include "loss/bce_loss.h"

#include "utils/utils.h"

void test_linear() {
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

    // Store original weights and biases
    float h_W_before[input_dim * output_dim], h_b_before[output_dim];
    CHECK_CUDA(cudaMemcpy(h_W_before, layer.get_weights(), sizeof(float) * input_dim * output_dim, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_before, layer.get_bias(), sizeof(float) * output_dim, cudaMemcpyDeviceToHost));

    // Backward pass, input gradient output
    layer.backward(d_dY, lr, batch_size);
    float* d_dX = layer.get_input_grad();

    // Get updated weights and biases
    float h_W_after[input_dim * output_dim], h_b_after[output_dim];
    CHECK_CUDA(cudaMemcpy(h_W_after, layer.get_weights(), sizeof(float) * input_dim * output_dim, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b_after, layer.get_bias(), sizeof(float) * output_dim, cudaMemcpyDeviceToHost));

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

    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dY);
    cudaFree(d_dX);
}

void test_relu() 
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
    print_h_matrix(h_output, 1, size);

    // Run backward
    relu.backward(d_grad_out, 0.0f, batch_size);
    d_grad_input = relu.get_input_grad();
    cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "ReLU backward grad_input:\n";
    print_h_matrix(h_grad_input, 1, size);

    std::cout << std::endl;

    // Free
    cudaFree(d_input);
    cudaFree(d_grad_out);
    cudaFree(d_grad_input);
}

void test_sigmoid()
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
    print_h_matrix(h_output, batch_size, dim);

    // Backward
    sigmoid.backward(d_output_grad, 0.0f, batch_size);
    d_input_grad = sigmoid.get_input_grad();

    float h_input_grad[size];
    cudaMemcpy(h_input_grad, d_input_grad, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sigmoid Backward d_input_grad: \n";
    print_h_matrix(h_input_grad, batch_size, dim);

    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output_grad);
    cudaFree(d_input_grad);
}

void test_bce_loss() {
    const int batch_size = 4;
    float h_yhat[batch_size] = { 0.9f, 0.2f, 0.7f, 0.1f };
    float h_y[batch_size] = { 1.0f, 0.0f, 1.0f, 0.0f };

    float* d_yhat, * d_y, * d_output_grad;
    cudaMalloc(&d_yhat, batch_size * sizeof(float));
    cudaMalloc(&d_y, batch_size * sizeof(float));

    cudaMemcpy(d_yhat, h_yhat, batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, batch_size * sizeof(float), cudaMemcpyHostToDevice);

    BCELoss loss;
    float avg_loss = loss.compute_loss(d_yhat, d_y, batch_size);
    std::cout << "[BCE Class] Average loss: " << avg_loss << std::endl;

    d_output_grad = loss.compute_loss_grad(d_yhat, d_y, batch_size);
    float h_output_grad[batch_size];
    cudaMemcpy(h_output_grad, d_output_grad, sizeof(float) * batch_size, cudaMemcpyDeviceToHost);

    std::cout << "[BCE Class] Gradients (dL/dYhat):" << std::endl;
    print_h_matrix(h_output_grad, batch_size, 1);

    std::cout << std::endl;

    cudaFree(d_yhat);
    cudaFree(d_y);
    cudaFree(d_output_grad);
}

void test_sequential(int n_epoch = 1000)
{
    const int input_dim = 2;
    const int hidden_dim = 3;
    const int output_dim = 1;
    const int batch_size = 4;
    const float learning_rate = 0.1f;

    float h_input[batch_size * input_dim] = {
        1.0f, 2.0f,
        2.0f, 1.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float h_labels[batch_size] = { 1, 0, 0, 1 };

    float* d_input;
    cudaMalloc(&d_input, sizeof(float) * input_dim * batch_size);
    cudaMemcpy(d_input, h_input, sizeof(float) * input_dim * batch_size, cudaMemcpyHostToDevice);

    float* d_labels;
    cudaMalloc(&d_labels, sizeof(float) * batch_size);
    cudaMemcpy(d_labels, h_labels, sizeof(float) * batch_size, cudaMemcpyHostToDevice);

    Sequential model;
    model.add_layer(new Linear(input_dim, hidden_dim, true));
    model.add_layer(new ReLU(hidden_dim));
    model.add_layer(new Linear(hidden_dim, output_dim));
    model.add_layer(new Sigmoid(output_dim));

    BCELoss loss;

    for (int i = 0; i < n_epoch; i++)
    {
        model.forward(d_input, batch_size);
        float* d_output = model.get_output();

        float h_output[output_dim * batch_size];
        cudaMemcpy(h_output, d_output, sizeof(float) * output_dim * batch_size, cudaMemcpyDeviceToHost);

        float loss_val = loss.compute_loss(d_output, d_labels, batch_size);
        float* d_loss_grad = loss.compute_loss_grad(d_output, d_labels, batch_size);

        model.backward(d_loss_grad, learning_rate, batch_size);

        cudaFree(d_loss_grad);

        std::cout << "[Epoch " << i + 1 << "] Loss: " << loss_val << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_labels);
}

int main()
{
    test_sequential();

    return 0;
}