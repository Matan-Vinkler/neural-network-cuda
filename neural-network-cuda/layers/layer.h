#ifndef LAYER_H_
#define LAYER_H_

#include <cuda_runtime.h>

class Layer
{
public:
    Layer() {}
	Layer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {}
	~Layer() { cudaFree(d_output); }

	virtual void forward(float* d_input, int batch_size) = 0;
	virtual void backward(float* d_output_grad, float learning_rate, int batch_size) = 0;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }

	float* get_output() const { return d_output; }
    float* get_input_grad() const { return d_input_grad; }

protected:
    int input_dim = 0;
    int output_dim = 0;

    float* d_input = nullptr;		// [ batch_size x input_dim ], cached for backprop
    float* d_output = nullptr;	    // [ batch_size x output_dim ]

    float* d_input_grad = nullptr;
};

#endif // !LAYER_H_

