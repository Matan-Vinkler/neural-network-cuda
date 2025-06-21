#include "data_load.h"

#include <fstream>
#include <sstream>
#include <iostream>

bool load_csv_data(const std::string& filename, std::vector<float>& input, std::vector<float>& labels, int input_dim)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int row_num = 0;

    while (std::getline(file, line))
    {
        if (row_num > 0)
        {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> row;

            while (std::getline(ss, token, ','))
            {
                row.push_back(std::stof(token));
            }

            if (row.size() != input_dim + 1)
            {
                std::cerr << "Row " << row_num << ", invalid row size: " << row.size() << ", expected" << input_dim + 1 << std::endl;
                return false;
            }

            labels.push_back(row[0]);
            for (int i = 1; i <= input_dim; i++)
            {
                input.push_back(row[i]);
            }
        }
        row_num++;
    }
}