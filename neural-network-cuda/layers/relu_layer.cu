#include "relu_layer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void relu_forward_kernel(float* input, float* output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		output[idx] = fmaxf(0.0f, input[idx]);
	}
}

__global__ void relu_backward_kernel(float* input, float* d_output_grad, float* d_input_grad, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		d_input_grad[idx] = input[idx] > 0 ? d_output_grad[idx] : 0;
	}
}

ReLU::ReLU(int input_dim) : Layer(input_dim, input_dim)
{
}

ReLU::~ReLU()
{
}

void ReLU::forward(float* d_input, int batch_size)
{
	this->d_input = d_input;
	int size = batch_size * input_dim;

	cudaFree(d_output);
	cudaMalloc(&d_output, sizeof(float) * size);

	int threads = 256;
	int blocks = (size + threads - 1) / threads;

	relu_forward_kernel << <blocks, threads >> > (d_input, d_output, size);

	cudaDeviceSynchronize();
}

void ReLU::backward(float* d_output_grad, float* d_input_grad, float, int batch_size)
{
	int size = batch_size * input_dim;

	int threads = 256;
	int blocks = (size + threads - 1) / threads;

	relu_backward_kernel << <blocks, threads >> > (d_input, d_output_grad, d_input_grad, size);

	cudaDeviceSynchronize();
}
