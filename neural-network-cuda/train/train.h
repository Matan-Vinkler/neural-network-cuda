#ifndef TRAIN_H_
#define TRAIN_H_

#include "../layers/sequential.h"
#include "../loss/bce_loss.h"

void train_model(Sequential& model, BCELoss& loss_fn, float* h_input, float* h_labels, int num_samples, int input_dim, int batch_size, int n_epoches, float lr, bool shuffle);

#endif // !TRAIN_H_
