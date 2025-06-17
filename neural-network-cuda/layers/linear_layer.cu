#include "linear_layer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <ctime>
#include <algorithm>

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

// C = A.dot(B)
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

// output = output + bias
__global__ void add_bias_kernel(float* output, float* bias, int batch_size, int output_dim)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < batch_size && col < output_dim)
	{
		output[row * output_dim + col] += bias[col];
	}
}

// dW = Xt * dY
__global__ void compute_dW_kernel(float* dW, float* X, float* dY, int batch_size, int input_dim, int output_dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < input_dim && col < output_dim)
	{
		// dW[row,col] = sum_i( X[i,row] * dY[i,col] )

		float val = 0.0f;
		for (int i = 0; i < batch_size; ++i)
		{
			val += X[i * input_dim + row] * dY[i * output_dim + col];
		}
		dW[row * output_dim + col] = val;
	}
}

// db = sum(dY, axis=0)
__global__ void compute_db_kernel(float* db, float* dY, int batch_size, int output_dim)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < output_dim)
	{
		// db[col] = sum_i( dY[i,col] )

		float val = 0.0f;
		for (int i = 0; i < batch_size; ++i)
		{
			val += dY[i * output_dim + col];
		}
		db[col] = val;
	}
}

// dX = dY * Wt
__global__ void compute_dX_kernel(float* dX, float* dY, float* W, int batch_size, int input_dim, int output_dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < batch_size && col < input_dim)
	{
		// dX[row,col] = sum_i( dY[row, i] * W[col, i] )

		float val = 0.0f;
		for (int i = 0; i < output_dim; i++)
		{
			val += dY[row * output_dim + i] * W[col * output_dim + i];
		}
		dX[row * input_dim + col] = val;
	}
}

// W -= lr * dW, b -= lr * db
__global__ void sgd_update_kernel(float* W, float* dW, float* b, float* db, float lr, int input_dim, int output_dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int weight_size = input_dim * output_dim;

	if (idx < weight_size)
	{
		W[idx] -= lr * dW[idx];
	}

	if (idx < output_dim)
	{
		b[idx] -= lr * db[idx];
	}
}

Linear::Linear(int input_dim, int output_dim) : Layer(input_dim, output_dim), d_weights(nullptr), d_bias(nullptr)
{
	this->input_dim = input_dim;
	this->output_dim = output_dim;
	this->d_input = nullptr;
	this->d_output = nullptr;

	cudaMalloc(&d_weights, sizeof(float) * input_dim * output_dim);
	cudaMalloc(&d_bias, sizeof(float) * output_dim);

	int weight_size = input_dim * output_dim;
	int bias_size = output_dim;

	int threads = 256;
	int weight_blocks = (weight_size + threads - 1) / threads;
	int bias_blocks = (bias_size + threads - 1) / threads;

	unsigned long seed = static_cast<unsigned long>(time(NULL));

	init_random_weights << <weight_blocks, threads >> > (d_weights, weight_size, 0.01f, seed);
	init_random_weights << <bias_blocks, threads >> > (d_bias, bias_size, 0.01f, seed + 1);
}

Linear::~Linear()
{
	cudaFree(d_weights);
	cudaFree(d_bias);
}

void Linear::forward(float* d_input, int batch_size)
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

	cudaDeviceSynchronize();
}

void Linear::backward(float* d_output_grad, float* d_input_grad, float learning_rate, int batch_size)
{
	int threads_dW = 16, threads_db = 256;
	dim3 blockDim(threads_dW, threads_dW);
	dim3 gridDimW((output_dim + threads_dW - 1) / threads_dW, (input_dim + threads_dW - 1) / threads_dW);
	dim3 gridDimX((input_dim + threads_dW - 1) / threads_dW, (batch_size + threads_dW - 1) / threads_dW);
	dim3 gridDimB((output_dim + threads_db - 1) / threads_db);

	float* d_dW, * d_db;
	cudaMalloc(&d_dW, sizeof(float) * input_dim * output_dim);
	cudaMalloc(&d_db, sizeof(float) * output_dim);

	compute_dW_kernel << < gridDimW, blockDim >> > (d_dW, d_input, d_output_grad, batch_size, input_dim, output_dim);
	compute_db_kernel << <gridDimB, threads_db >> > (d_db, d_output_grad, batch_size, output_dim);
	compute_dX_kernel << <gridDimX, blockDim >> > (d_input_grad, d_output_grad, d_weights, batch_size, input_dim, output_dim);

	int max_dim = std::max(input_dim * output_dim, output_dim);
	sgd_update_kernel << <(max_dim + threads_db - 1) / threads_db, threads_db >> > (d_weights, d_dW, d_bias, d_db, learning_rate, input_dim, output_dim);

	cudaFree(d_dW);
	cudaFree(d_db);

	cudaDeviceSynchronize();
}
