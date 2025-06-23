#ifndef BIN_ACC_H_
#define BIN_ACC_H_

class BinaryAccuracy
{
public:
    BinaryAccuracy(float threshold);

    float calculate_acc(float* d_preds, float* d_labels, int batch_size);

private:
    float threshold;
};

#endif // !BIN_ACC_H_
