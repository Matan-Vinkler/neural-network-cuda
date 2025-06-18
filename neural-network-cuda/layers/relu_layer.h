#ifndef RELU_LAYER_H_
#define RELU_LAYER_H_

#include "layer.h"

class ReLU : public Layer
{
public:
	ReLU(int input_dim);
	~ReLU();

	void forward(float* d_input, int batch_size);
	void backward(float* d_output_grad, float learning_rate, int batch_size);
};

#endif // !RELU_LAYER_H_
