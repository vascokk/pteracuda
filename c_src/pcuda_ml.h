#ifndef PCUDA_ML
#define PCUDA_ML

#include <vector>

void pcuda_gd(std::vector<double> *theta, std::vector<double> *x, std::vector<double> *y, const unsigned int num_features, const unsigned int num_samples);
void pcuda_gd_learn(std::vector<double> *theta, std::vector<double> *x, std::vector<double> *y, const unsigned int num_features, const unsigned int num_samples, const float learning_rate, const unsigned int iterations);

#endif