#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_

#include "layer.h"

#include <vector>

#include "../loss/bce_loss.h"

class Sequential : public Layer
{
public:
    Sequential() {}
    ~Sequential();

    void add_layer(Layer* layer);

    void forward(float* d_input, int batch_size);
    void backward(float* d_output_grad, float learning_rate, int batch_size);

private:
    std::vector<Layer*> layers_vec;
};

#endif // !SEQUENTIAL_H_
