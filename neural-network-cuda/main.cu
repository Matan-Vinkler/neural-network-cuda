#include <iostream>

#include "layers/linear_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/sequential.h"
#include "loss/bce_loss.h"
#include "utils/utils.h"
#include "train/train.h"
#include "data/data_load.h"

void test_train_model()
{
    const int input_dim = 2;
    const int hidden_dim = 3;
    const int output_dim = 1;
    const int batch_size = 4;
    const int num_samples = 4;
    const float learning_rate = 0.1f;
    const int epoches = 1000;

    float h_input[batch_size * input_dim] = {
        1.0f, 2.0f,
        2.0f, 1.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    float h_labels[batch_size] = { 1, 0, 0, 1 };

    Sequential model;
    model.add_layer(new Linear(input_dim, hidden_dim, true));
    model.add_layer(new ReLU(hidden_dim));
    model.add_layer(new Linear(hidden_dim, output_dim));
    model.add_layer(new Sigmoid(output_dim));

    BCELoss loss;

    train_model(model, loss, h_input, h_labels, num_samples, input_dim, batch_size, epoches, learning_rate, true);
}

void test_data_load()
{
    std::vector<float> vec_inputs;
    std::vector<float> vec_labels;

    int input_dim = 64 * 64;
    if (!load_csv_data("data/data.csv", vec_inputs, vec_labels, input_dim))
    {
        std::cerr << "Failed to load data!" << std::endl;
        return;
    }

    int num_samples = vec_labels.size();

    print_dataset(vec_inputs, vec_labels, input_dim, num_samples);
}

int main()
{
    test_data_load();

    for (int i = 0; i < 10; i++)
    {
        std::cout << "[----------------------------------------------------Test " << i + 1 << "----------------------------------------------------]\n";
        test_train_model();
        std::cout << "[--------------------------------------------------------------------------------------------------------------]" << std::endl;
    }

    //TODO: Fix loss stuck value (maybe gradient vanishing)

    return 0;
}