#include "sigmoid_layer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void sigmoid_forward_kernel(float* input, float* output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		output[idx] = 1.0f / (1.0f + expf(-input[idx]));
	}
}

__global__ void sigmoid_backward_kernel(float* output, float* d_output_grad, float* d_input_grad, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		float y = output[idx];
		d_input_grad[idx] = d_output_grad[idx] * y * (1.0f - y);
	}
}

Sigmoid::Sigmoid(int input_dim) : Layer(input_dim, input_dim)
{
}

Sigmoid::~Sigmoid()
{
}

void Sigmoid::forward(float* d_input, int batch_size)
{
	this->d_input = d_input;
	int size = batch_size * input_dim;

	cudaFree(d_output);
	cudaMalloc(&d_output, sizeof(float) * size);

	int threads = 256;
	int blocks = (size + threads - 1) / threads;

	sigmoid_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);

	cudaDeviceSynchronize();
}

void Sigmoid::backward(float* d_output_grad, float, int batch_size)
{
	int size = batch_size * input_dim;

    cudaFree(d_input_grad);
    cudaMalloc(&d_input_grad, sizeof(float) * size);

	int threads = 256;
	int blocks = (size + threads - 1) / threads;

	sigmoid_backward_kernel<<<blocks, threads>>>(d_output, d_output_grad, d_input_grad, size);

	cudaDeviceSynchronize();
}
