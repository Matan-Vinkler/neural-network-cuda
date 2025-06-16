#ifndef LINEAR_LAYER_H_
#define LINEAR_LAYER_H_

class LinearLayer
{
public:
    int input_dim;
    int output_dim;

    float* d_weights;   // [ input_dim x output_dim ]
    float* d_bias;      // [ 1 x output_dim ]

    float* d_input;     // [ batch_size x dim_input ], cached for backprop
    float* d_output;    // [ batch_size x dim_output ]

    LinearLayer(int input_dim, int output_dim);
    ~LinearLayer();

    void forward(float* d_input, int batch_size);
    void backward(float* d_output_grad, float* d_input_grad, float learning_rate, int batch_size);
};

#endif // !LINEAR_LAYER_H_

