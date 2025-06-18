#ifndef LINEAR_LAYER_H_
#define LINEAR_LAYER_H_

#include "layer.h"

class Linear : public Layer
{
public:
    float* d_weights;   // [ input_dim x output_dim ]
    float* d_bias;      // [ 1 x output_dim ]

    Linear(int input_dim, int output_dim);
    ~Linear();

    void forward(float* d_input, int batch_size);
    void backward(float* d_output_grad, float learning_rate, int batch_size);
};

#endif // !LINEAR_LAYER_H_

