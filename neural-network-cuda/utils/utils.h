#ifndef UTILS_H_
#define UTILS_H_

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

void print_h_matrix(const float* data, int rows, int cols);

void print_d_matrix(const float* d_data, int rows, int cols);

#endif // !UTILS_H_
