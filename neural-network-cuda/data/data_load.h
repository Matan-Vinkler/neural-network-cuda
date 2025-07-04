#ifndef DATA_LOAD_H_
#define DATA_LOAD_H_

#include <string>
#include <vector>

bool load_csv_data(const std::string& filename, std::vector<float>& input, std::vector<float>& labels, int input_dim);

void normalize_data(std::vector<float>& inputs);

#endif // !DATA_LOAD_H_
