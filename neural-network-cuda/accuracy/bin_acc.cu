#include "bin_acc.h"

// Solves a weired bug that the highlighter doesn't recognize 'atomicAdd'
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void compute_accuracy_kernel(float* d_preds, float* d_labels, int size, int* d_correct_count, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int pred = (d_preds[idx] >= threshold) ? 1 : 0;
        int label = (d_labels[idx] >= threshold) ? 1 : 0;

        if (pred == label)
        {
            atomicAdd(d_correct_count, 1);
        }
    }
}

BinaryAccuracy::BinaryAccuracy(float threshold) : threshold(threshold)
{
}

float BinaryAccuracy::calculate_acc(float* d_preds, float* d_labels, int batch_size)
{
    int* d_correct_count;
    int h_correct_count;

    cudaMalloc(&d_correct_count, sizeof(float));
    cudaMemset(d_correct_count, 0, sizeof(float));

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    compute_accuracy_kernel << <blocks, threads >> > (d_preds, d_labels, batch_size, d_correct_count, threshold);

    cudaMemcpy(&h_correct_count, d_correct_count, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_correct_count);

    return static_cast<float>(h_correct_count) / batch_size;
}
