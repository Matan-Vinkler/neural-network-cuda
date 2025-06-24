#include "validate.h"

#include <cuda_runtime.h>
#include <iostream>

void validate_model(Sequential& model, BinaryAccuracy& acc_fn, float* h_inputs, float* h_labels, int num_samples, int input_dim, int batch_size)
{
    std::cout << "Begin validating..." << std::endl;

    float* d_inputs, * d_labels;
    cudaMalloc(&d_inputs, sizeof(float) * batch_size * input_dim);
    cudaMalloc(&d_labels, sizeof(float) * batch_size);

    int num_batches = num_samples / batch_size;
    float total_acc = 0.0f;

    for (int batch = 0; batch < num_batches; batch++)
    {
        float* h_input_batch = h_inputs + batch * batch_size * input_dim;
        float* h_label_batch = h_labels + batch * batch_size;

        cudaMemcpy(d_inputs, h_input_batch, sizeof(float) * batch_size * input_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, h_label_batch, sizeof(float) * batch_size, cudaMemcpyHostToDevice);

        model.forward(d_inputs, batch_size);
        float* d_outputs = model.get_output();

        float acc_val = acc_fn.calculate_acc(d_outputs, d_labels, batch_size);
        total_acc += acc_val;

        std::cout << "[Batch " << batch + 1 << "] Accuracy: " << acc_val << "\n";
    }

    float avg_acc = total_acc / num_batches;

    std::cout << "Average accuracy: " << avg_acc << "\nValidating complete!" << std::endl;

    cudaFree(d_inputs);
    cudaFree(d_labels);
}
