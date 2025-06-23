#include "train.h"

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <random>
#include <cstring>
#include <iostream>

void shuffle_dataset(float* h_input, float* h_labels, int num_samples, int input_dim)
{
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Create full sample pairs: each has input + label
    std::vector<std::vector<float>> input_samples(num_samples, std::vector<float>(input_dim));
    std::vector<float> label_samples(num_samples);

    // Copy original data into structured containers
    for (int i = 0; i < num_samples; ++i)
    {
        std::memcpy(input_samples[i].data(), &h_input[i * input_dim], input_dim * sizeof(float));
        label_samples[i] = h_labels[i];
    }

    // Shuffle according to indices
    for (int i = 0; i < num_samples; ++i)
    {
        std::memcpy(&h_input[i * input_dim], input_samples[indices[i]].data(), input_dim * sizeof(float));
        h_labels[i] = label_samples[indices[i]];
    }
}

void train_model(Sequential& model, BCELoss& loss_fn, BinaryAccuracy& acc_fn, float* h_input, float* h_labels, int num_samples, int input_dim, int batch_size, int n_epoches, float lr, bool shuffle = true)
{
    std::cout << "Begin training..." << std::endl;

    float* d_input, * d_labels;
    cudaMalloc(&d_input, sizeof(float) * batch_size * input_dim);
    cudaMalloc(&d_labels, sizeof(float) * batch_size);

    for (int i_epoch = 0; i_epoch < n_epoches; i_epoch++)
    {
        if (shuffle)
        {
            shuffle_dataset(h_input, h_labels, num_samples, input_dim);
        }

        float total_loss = 0.0f;
        float total_acc = 0.0f;
        int num_batches = num_samples / batch_size;

        for (int i_batch = 0; i_batch < num_batches; i_batch++)
        {
            float* h_input_batch = h_input + i_batch * batch_size * input_dim;
            float* h_label_batch = h_labels + i_batch * batch_size;

            cudaMemcpy(d_input, h_input_batch, sizeof(float) * batch_size * input_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, h_label_batch, sizeof(float) * batch_size, cudaMemcpyHostToDevice);

            model.forward(d_input, batch_size);
            float* d_output = model.get_output();

            float loss = loss_fn.compute_loss(d_output, d_labels, batch_size);
            total_loss += loss;

            float acc = acc_fn.calculate_acc(d_output, d_labels, batch_size);
            total_acc += acc;

            float* d_loss_grad = loss_fn.compute_loss_grad(d_output, d_labels, batch_size);
            model.backward(d_loss_grad, lr, batch_size);
        }

        float avg_loss = total_loss / num_batches;
        float avg_acc = total_acc / num_batches;
        if (i_epoch % 10 == 0 || i_epoch == n_epoches - 1)
        {
            std::cout << "[Epoch " << i_epoch + 1 << "] Loss: " << avg_loss << ", Accuracy: " << avg_acc << "\n";
        }
    }

    cudaFree(d_input);
    cudaFree(d_labels);

    std::cout << "Training complete!" << std::endl;
}
