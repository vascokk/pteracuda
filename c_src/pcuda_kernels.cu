#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/set_operations.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cublas.h>

#include <iostream>
#include <cmath>
#include <vector>

template <typename T>
struct sigmoid{
	__host__ __device__ T operator()(const T &x){return 1 / (1 + exp(-x));}
};

template <typename T>
struct sigmoid2{
  __host__ __device__ T operator()(const T &x){return tanh(x);}
};


void pcuda_sigmoid(std::vector<double> *a, std::vector<double> *b){
  thrust::device_vector<float> d_a = *a;
  thrust::device_vector<float> d_b(b->size());

  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), sigmoid<float>());

  thrust::copy(d_b.begin(), d_b.end(), b->begin());
}

void pcuda_tanh(std::vector<double> *a, std::vector<double> *b){
  thrust::device_vector<float> d_a = *a;
  thrust::device_vector<float> d_b(b->size());

  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), sigmoid2<float>());

  thrust::copy(d_b.begin(), d_b.end(), b->begin());
}
