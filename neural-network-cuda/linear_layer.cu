#include "linear_layer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <ctime>

// This function init vector or matrix with random values
__global__ void init_random_weights(float* data, int size, float scale, unsigned long seed)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		curandState state;
		curand_init(seed, i, 0, &state);

		float r = curand_uniform(&state);
		data[i] = (r - 0.5f) * scale;
	}
}

// This function performs C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N)
	{
		float sum = 0.0f;
		for (int i = 0; i < K; i++)
		{
			sum += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] = sum;
	}
}

__global__ void add_bias_kernel(float* output, float* bias, int batch_size, int output_dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < batch_size && col < output_dim)
	{
		output[row * output_dim + col] += bias[col];
	}
}

LinearLayer::LinearLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim), d_weights(nullptr), d_bias(nullptr), d_output(nullptr), d_input(nullptr)
{
	cudaMalloc(&d_weights, sizeof(float) * input_dim * output_dim);
	cudaMalloc(&d_bias, sizeof(float) * output_dim);

	int weight_size = input_dim * output_dim;
	int bias_size = output_dim;

	int threads = 256;
	int weight_blocks = (weight_size + threads - 1) / threads;
	int bias_blocks = (bias_size + threads - 1) / threads;

	unsigned long seed = time(NULL);

	init_random_weights << <weight_blocks, threads >> > (d_weights, weight_size, 0.01f, seed);
	init_random_weights << <bias_blocks, threads >> > (d_bias, bias_size, 0.01f, seed + 1);
}

LinearLayer::~LinearLayer()
{
	cudaFree(d_weights);
	cudaFree(d_bias);
	cudaFree(d_output);
}

void LinearLayer::forward(float* d_input, int batch_size)
{
	this->d_input = d_input; // save cache for backprop

	cudaFree(d_output);
	cudaMalloc(&d_output, sizeof(float) * batch_size * output_dim);

	// d_output = d_input * d_weights
	dim3 threads_w(16, 16);
	dim3 blocks_w((output_dim + 15) / 16, (batch_size + 15) / 16);
	matmul_kernel << <blocks_w, threads_w >> > (d_input, d_weights, d_output, batch_size, input_dim, output_dim);

	// d_output = d_output + d_bias
	dim3 threads_b(16, 16);
	dim3 blocks_b((batch_size + 15) / 16, (output_dim + 15) / 16);
	add_bias_kernel << <blocks_b, threads_b >> > (d_output, d_bias, batch_size, output_dim);
}

void LinearLayer::backward(float* d_output_grad, float* d_input_grad, float learning_rate, int batch_size)
{
	//TODO: Implement 'backward'
}
