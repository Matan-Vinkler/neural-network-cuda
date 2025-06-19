#ifndef SEQUENTIAL_H_
#define SEQUENCTIAL_H_

#include "layer.h"

#include <vector>

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
