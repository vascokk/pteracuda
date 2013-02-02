#ifndef PCUDA_KERNELS
#define PCUDA_KERNELS

#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

template <typename T>
struct sigmoid{
	__host__ __device__ T operator()(const T &x){return 1 / (1 + exp(-x));}
};

template <typename T>
struct sigmoid2{
  __host__ __device__ T operator()(const T &x){return tanh(x);}
};

template <typename T>
struct log_func{
  __host__ __device__ T operator()(const T &x){return log(x);}
};


void pcuda_sigmoid(std::vector<double> *a, std::vector<double> *b);
void pcuda_tanh(std::vector<double> *a, std::vector<double> *b);
void pcuda_log(std::vector<double> *a, std::vector<double> *b);

#endif