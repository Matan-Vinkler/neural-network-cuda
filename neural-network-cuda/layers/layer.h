#ifndef LAYER_H_
#define LAYER_H_

class Layer
{
public:
	int input_dim;
	int output_dim;

	float* d_input;		// [ batch_size x input_dim ], cached for backprop
	float* d_output;	// [ batch_size x output_dim ]

    float* d_input_grad;

	Layer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim), d_input(nullptr), d_output(nullptr), d_input_grad(nullptr) {}
	~Layer() { cudaFree(d_output); }

	virtual void forward(float* d_input, int batch_size) = 0;
	virtual void backward(float* d_output_grad, float learning_rate, int batch_size) = 0;

	float* get_output() const { return d_output; }
    float* get_input_grad() const { return d_input_grad; }
};

#endif // !LAYER_H_

