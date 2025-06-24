#include <iostream>

#include "layers/linear_layer.h"
#include "layers/relu_layer.h"
#include "layers/sigmoid_layer.h"
#include "layers/sequential.h"
#include "loss/bce_loss.h"
#include "accuracy/bin_acc.h"
#include "utils/utils.h"
#include "train/train.h"
#include "validate/validate.h"
#include "data/data_load.h"

int main()
{
    const int input_dim = 64 * 64;
    const int hidden_dim = 1024;
    const int output_dim = 1;
    const int batch_size = 10;
    const float learning_rate = 0.01f;
    const int epoches = 100;

    // ---- Load training data ----
    std::vector<float> vec_train_inputs, vec_train_labels;
    if (!load_csv_data("data/train_data.csv", vec_train_inputs, vec_train_labels, input_dim))
    {
        std::cerr << "Failed to load train data!" << std::endl;
        return -1;
    }
    normalize_data(vec_train_inputs);

    int num_samples_train = static_cast<int>(vec_train_labels.size());
    float* h_train_inputs = vec_train_inputs.data();
    float* h_train_labels = vec_train_labels.data();

    // ---- Build the model ----
    Sequential model;
    model.add_layer(new Linear(input_dim, hidden_dim));
    model.add_layer(new ReLU(hidden_dim));
    model.add_layer(new Linear(hidden_dim, output_dim));
    model.add_layer(new Sigmoid(output_dim));

    BCELoss loss_fn;
    BinaryAccuracy acc_fn(0.5);

    // ---- Train the model ----
    train_model(model, loss_fn, acc_fn, h_train_inputs, h_train_labels, num_samples_train, input_dim, batch_size, epoches, learning_rate, true);

    // ---- Load validation data ----
    std::vector<float> vec_val_inputs, vec_val_labels;
    if (!load_csv_data("data/val_data.csv", vec_val_inputs, vec_val_labels, input_dim))
    {
        std::cerr << "Failed to load validate data!" << std::endl;
        return -1;
    }
    normalize_data(vec_val_inputs);

    int num_samples_val = static_cast<int>(vec_val_labels.size());

    float* h_val_inputs = vec_val_inputs.data();
    float* h_val_labels = vec_val_labels.data();

    // ---- Validate model ----
    validate_model(model, acc_fn, h_val_inputs, h_val_labels, num_samples_val, input_dim, batch_size);

    return 0;
}