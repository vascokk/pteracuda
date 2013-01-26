#ifndef PCUDA_KERNELS
#define PCUDA_KERNELS

#include <vector>

void pcuda_sigmoid(std::vector<double> *a, std::vector<double> *b);
void pcuda_tanh(std::vector<double> *a, std::vector<double> *b);
void pcuda_log(std::vector<double> *a, std::vector<double> *b);

#endif