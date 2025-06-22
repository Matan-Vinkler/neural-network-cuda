#include <iostream>

#include "layers/linear_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/sequential.h"
#include "loss/bce_loss.h"
#include "utils/utils.h"
#include "train/train.h"
#include "data/data_load.h"

int main()
{
    int input_dim = 64 * 64;
    int hidden_dim = 1024;
    int output_dim = 1;

    std::vector<float> vec_inputs;
    std::vector<float> vec_labels;

    if (!load_csv_data("data/data.csv", vec_inputs, vec_labels, input_dim))
    {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }

    normalize_data(vec_inputs);

    int num_samples = vec_labels.size();
    int batch_size = 10;
    const float learning_rate = 0.01f;
    const int epoches = 1000;

    float* h_inputs = vec_inputs.data();
    float* h_labels = vec_labels.data();

    Sequential model;
    model.add_layer(new Linear(input_dim, hidden_dim));
    model.add_layer(new ReLU(hidden_dim));
    model.add_layer(new Linear(hidden_dim, output_dim));
    model.add_layer(new Sigmoid(output_dim));

    BCELoss loss_fn;

    train_model(model, loss_fn, h_inputs, h_labels, num_samples, input_dim, batch_size, epoches, learning_rate, true);

    return 0;

    //TODO: Add test data routine
}