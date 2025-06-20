#include "sequential.h"

Sequential::~Sequential()
{
    for (Layer* layer : layers_vec)
    {
        delete layer;
    }
}

void Sequential::add_layer(Layer* layer)
{
    if (layers_vec.empty())
    {
        this->input_dim = layer->get_input_dim();
    }

    layers_vec.push_back(layer);

    this->output_dim = layer->get_output_dim();
}

void Sequential::forward(float* d_input, int batch_size)
{
    float* current_input = d_input;

    for (Layer* layer : layers_vec)
    {
        layer->forward(current_input, batch_size);
        current_input = layer->get_output();
    }

    d_output = current_input;
}

void Sequential::backward(float* d_output_grad, float learning_rate, int batch_size)
{
    float* current_output_grad = d_output_grad;

    for (auto it = layers_vec.rbegin(); it != layers_vec.rend(); ++it)
    {
        Layer* layer = *it;
        layer->backward(current_output_grad, learning_rate, batch_size);
        current_output_grad = layer->get_input_grad();
    }

    d_input_grad = current_output_grad;
}
