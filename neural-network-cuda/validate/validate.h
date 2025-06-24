#ifndef VALIDATE_H_
#define VALIDATE_H_

#include "../layers/sequential.h"
#include "../accuracy/bin_acc.h"

void validate_model(Sequential& model, BinaryAccuracy& acc_fn, float* h_inputs, float* h_labels, int num_samples, int input_dim, int batch_size);

#endif // !VALIDATE_H_
