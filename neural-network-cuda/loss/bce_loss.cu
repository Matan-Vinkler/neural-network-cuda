#include "bce_loss.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void bce_loss_kernel(const float* yhat, const float* y, float* losses, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float p = yhat[idx];
        float label = y[idx];

        p = fminf(fmaxf(p, 1e-7f), 1.0f - 1e-7f);

        losses[idx] = -(label * logf(p) + (1.0f - label) * logf(1.0f - p));
    }
}

__global__ void bce_grad_kernel(const float* yhat, const float* y, float* grad, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float p = yhat[idx];
        float label = y[idx];

        p = fminf(fmaxf(p, 1e-7f), 1.0f - 1e-7f);

        grad[idx] = (p - label) / (p * (1.0f - p));
    }
}

float BCELoss::compute_loss(float* d_yhat, float* d_y, int batch_size)
{
    float* d_losses;
    cudaMalloc(&d_losses, sizeof(float) * batch_size);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    bce_loss_kernel << <blocks, threads >> > (d_yhat, d_y, d_losses, batch_size);

    float* h_losses = new float[batch_size];
    cudaMemcpy(h_losses, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_loss = 0.0f;
    for (int i = 0; i < batch_size; i++)
    {
        sum_loss += h_losses[i];
    }
    float avg_loss = sum_loss / batch_size;

    cudaFree(d_losses);
    delete[] h_losses;

    cudaDeviceSynchronize();

    return avg_loss;
}

float* BCELoss::compute_loss_grad(float* d_yhat, float* d_y, int batch_size)
{
    float* d_grad;
    cudaMalloc(&d_grad, sizeof(float) * batch_size);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    bce_grad_kernel << <blocks, threads >> > (d_yhat, d_y, d_grad, batch_size);
    cudaDeviceSynchronize();

    return d_grad;
}
