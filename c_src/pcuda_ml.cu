   

#include "cuda.h"

#include "cublas_v2.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <iostream>

#include "pcuda_kernels.h"
#include "pcuda_blas.h"

void gradient_descent(cublasHandle_t handle, thrust::device_vector<float> &d_theta, thrust::device_vector<float> &d_x, thrust::device_vector<float> &d_y, const unsigned int num_features, const unsigned int num_samples)
{
    int lda,ldb,ldc;

    // Create a handle for CUBLAS
    cublasStatus_t res;

    //Grad = (1/m)* ( X * (sigmoid(Theta*X) - Y) )
    // tmp1 = gemm(1*Theta*X + 0*H)
    lda=1,ldb=num_samples, ldc=1;
    thrust::device_vector<float> d_tmp1(num_samples, 0.0);
    const float alf = 1.0;
    const float bet = 0.0;
    const float *_alpha = &alf;
    const float *_beta =  &bet;
    res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, num_samples, num_features, _alpha, thrust::raw_pointer_cast(&d_theta[0]), lda, thrust::raw_pointer_cast(&d_x[0]), ldb, _beta, thrust::raw_pointer_cast(&d_tmp1[0]), ldc);

    //% H=sigmoid(Theta*X)
    thrust::device_vector<float> d_h(d_y.size());
    thrust::transform(d_tmp1.begin(), d_tmp1.end(), d_h.begin(), sigmoid<float>());

    //%H - Y
    thrust::transform(d_y.begin(), d_y.end(), d_h.begin(), d_h.begin(), saxpy_functor<float>(-1.0));
    
    const float alf2 = 1.0/num_samples;
    const float *alpha2 = &alf2;
    res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_features, num_samples, alpha2, thrust::raw_pointer_cast(&d_h[0]), lda, thrust::raw_pointer_cast(&d_x[0]), ldb, _beta, thrust::raw_pointer_cast(&d_theta[0]), ldc);
}


void pcuda_gd(std::vector<double> *theta, std::vector<double> *x, std::vector<double> *y, const unsigned int num_features, const unsigned int num_samples)
{
    int lda,ldb,ldc;

    thrust::device_vector<float> d_theta = *theta;
    thrust::device_vector<float> d_x = *x;
    thrust::device_vector<float> d_y = *y;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    gradient_descent(handle, d_theta, d_x, d_y, num_features, num_samples);

    thrust::copy(d_theta.begin(), d_theta.end(), theta->begin());
    // Destroy the handle
    cublasDestroy(handle);

}


void pcuda_gd_learn(std::vector<double> *theta, std::vector<double> *x, std::vector<double> *y, const unsigned int num_features, const unsigned int num_samples, const float learning_rate, const unsigned int iterations){
    cublasHandle_t handle;
    cublasCreate(&handle);

    thrust::device_vector<float> d_theta = *theta;
    thrust::device_vector<float> d_x = *x;
    thrust::device_vector<float> d_y = *y;
    thrust::device_vector<float> d_theta_tmp = d_theta;

    for(int i=0; i<iterations; i++){
        gradient_descent(handle, d_theta, d_x, d_y, num_features, num_samples);
        thrust::transform(d_theta.begin(), d_theta.end(), d_theta_tmp.begin(), d_theta_tmp.begin(), saxpy_functor<float>(-learning_rate));
        d_theta = d_theta_tmp;
    }
    thrust::copy(d_theta.begin(), d_theta.end(), theta->begin());

    // Destroy the handle
    cublasDestroy(handle);
}