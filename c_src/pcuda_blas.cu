#include <vector>
#include <stdio.h>
#include <iostream>
#include <algorithm>

#include "cuda.h"
#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/set_operations.h>
#include <thrust/extrema.h>

struct CastToFloat
{
    float operator()(double value) const { return static_cast<float>(value);}
};

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void pcuda_gemm(const int transpose_a, const int transpose_b, const int m, const int n, const int k, const double alpha, std::vector<double> *a, std::vector<double> *b, const double beta, std::vector<double> *c){
    int lda=m,ldb=k,ldc=m;
    const float alf = (float)alpha;
    const float bet = (float)beta;
    const float *_alpha = &alf;
    const float *_beta =  &bet;
    cublasOperation_t _transpose_a, _transpose_b;

    switch (transpose_a){
        case 0: 
            _transpose_a = CUBLAS_OP_N;
            break;
        case 1: 
            _transpose_a = CUBLAS_OP_T;
            lda = k;
            break;
        case 2: 
            _transpose_a = CUBLAS_OP_C;
            lda = k;
            break;
    }

    switch (transpose_b){
        case 0: 
            _transpose_b = CUBLAS_OP_N;
            break;
        case 1: 
            _transpose_b = CUBLAS_OP_T;
            ldb = n;
            break;
        case 2: 
            _transpose_b = CUBLAS_OP_C;
            ldb = n;
            break;
    }

    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a = *a;
    thrust::device_vector<float> d_b = *b;
    thrust::device_vector<float> d_c = *c;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = cublasSgemm(handle, _transpose_a, _transpose_b, m, n, k, _alpha, thrust::raw_pointer_cast(&d_a[0]), lda, thrust::raw_pointer_cast(&d_b[0]), ldb, _beta, thrust::raw_pointer_cast(&d_c[0]), ldc);
    //std::cout << "\ncublasSgemm Status = " << res << std::endl;

    thrust::copy(d_c.begin(), d_c.end(), c->begin());
    // Destroy the handle
    cublasDestroy(handle);
}

void pcuda_gemv(const int transpose, const int m, const int n, const double alpha, std::vector<double> *a, std::vector<double> *x, const double beta, std::vector<double> *y){
    int lda=m;
    const float alf = (float)alpha;
    const float bet = (float)beta;
    const float *_alpha = &alf;
    const float *_beta =  &bet;
    int incx=1, incy=1;

    cublasOperation_t _transpose;

    switch (transpose){
        case 0: _transpose = CUBLAS_OP_N;break;
        case 1: _transpose = CUBLAS_OP_T;break;
        case 2: _transpose = CUBLAS_OP_C;break;
    }

    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a = *a;
    thrust::device_vector<float> d_x = *x;
    thrust::device_vector<float> d_y = *y;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = cublasSgemv(handle, _transpose, m, n, _alpha, thrust::raw_pointer_cast(&d_a[0]), lda, thrust::raw_pointer_cast(&d_x[0]), incx, _beta, thrust::raw_pointer_cast(&d_y[0]), incy);
    //std::cout << "\ncublasSgemv Status = " << res << std::endl;

    thrust::copy(d_y.begin(), d_y.end(), y->begin());

    // Destroy the handle
    cublasDestroy(handle);
}

template <typename T>
struct saxpy_functor
{
    const T a;

    saxpy_functor(T _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x, const T& y) const { 
            return a * x + y;
        }
};

//SAXPY:  y <- a * x + y
void pcuda_saxpy(double a, std::vector<double> *x, std::vector<double> *y)
{
    
    const float _a = (float)a;
    thrust::device_vector<float> d_x = *x;
    thrust::device_vector<float> d_y = *y;

    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), saxpy_functor<float>(_a));

    thrust::copy(d_y.begin(), d_y.end(), y->begin());
}

//Transpose:  B<-A'
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

void pcuda_transpose(const int _m, const int _n, std::vector<double> *a, std::vector<double> *b){

    size_t m = _m; 
    size_t n = _n;
    
    thrust::device_vector<float> d_a = *a;
    thrust::device_vector<float> d_b(b->size());

    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
        (thrust::make_transform_iterator(indices, transpose_index(n, m)),
        thrust::make_transform_iterator(indices, transpose_index(n, m)) + d_b.size(),
        d_a.begin(),
        d_b.begin());    

    thrust::copy(d_b.begin(), d_b.end(), b->begin());
}

void pcuda_geam(const int transpose_a, const int transpose_b, const int m, const int n, const double alpha, std::vector<double> *a, const double beta, std::vector<double> *b, std::vector<double> *c){
    int lda=m,ldb=m,ldc=m;
    const float alf = (float)alpha;
    const float bet = (float)beta;
    const float *_alpha = &alf;
    const float *_beta =  &bet;
    cublasOperation_t _transpose_a, _transpose_b;

    switch (transpose_a){
        case 0: _transpose_a = CUBLAS_OP_N;break;
        case 1: _transpose_a = CUBLAS_OP_T;break;
        case 2: _transpose_a = CUBLAS_OP_C;break;
    }

    switch (transpose_b){
        case 0: _transpose_b = CUBLAS_OP_N;break;
        case 1: _transpose_b = CUBLAS_OP_T;break;
        case 2: _transpose_b = CUBLAS_OP_C;break;
    }

    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a = *a;
    thrust::device_vector<float> d_b = *b;
    thrust::device_vector<float> d_c(c->size());

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = cublasSgeam(handle, _transpose_a, _transpose_b, m, n, _alpha, thrust::raw_pointer_cast(&d_a[0]), lda, _beta, thrust::raw_pointer_cast(&d_b[0]), ldb, thrust::raw_pointer_cast(&d_c[0]), ldc);
    //std::cout << "\ncublasSgemm Status = " << res << std::endl;

    thrust::copy(d_c.begin(), d_c.end(), c->begin());
    // Destroy the handle
    cublasDestroy(handle);
}


template <typename T>
struct smm_functor
{
    const T a;

    smm_functor(T _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x) const { 
            return a * x;
        }
};

void pcuda_smm(const double alpha, std::vector<double> *a, std::vector<double> *b){
    const float _alpha = (float)alpha;
    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a = *a;
    thrust::device_vector<float> d_b(b->size());

    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), smm_functor<float>(_alpha));

    thrust::copy(d_b.begin(), d_b.end(), b->begin());
      
}
