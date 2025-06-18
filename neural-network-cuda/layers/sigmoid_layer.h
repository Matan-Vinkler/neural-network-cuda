#ifndef SIGMOID_LAYER_H_
#define SIGMOID_LAYER_H_

#include "layer.h"

class Sigmoid : public Layer
{
public:
	Sigmoid(int input_dim);
	~Sigmoid();

	void forward(float* d_input, int batch_size);
	void backward(float* d_output_grad, float learning_rate, int batch_size);
};

#endif // !SIGMOID_LAYER_H_
