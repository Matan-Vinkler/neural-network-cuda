#ifndef BCE_LOSS_H_
#define BCE_LOSS_H_

class BCELoss
{
public:
    float compute_loss(float* d_yhat, float* d_y, int batch_size);
    float* compute_loss_grad(float* d_yhat, float* d_y, int batch_size);
};

#endif // BCE_LOSS_H_