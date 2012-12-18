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

#include "pcuda_string.h"

PCudaString::PCudaString() {
    this->len = -1;
    this->str = NULL;
}

PCudaString::PCudaString(const std::string& other) {
    this->len = other.length();
    this->ptr = thrust::device_malloc<char>(this->len + 1);
    this->str = raw_pointer_cast(this->ptr);
    cudaMemcpy(this->str, other.c_str(), this->len, cudaMemcpyHostToDevice);
}

PCudaString::PCudaString(const PCudaString& other) {
    this->len = other.len;
    this->str = other.str;
    this->ptr = other.ptr;
}

int PCudaString::length() {
    return this->len;
}

int PCudaString::cstr_length() {
    return this->len + 1;
}

PCudaString::operator std::string() {
    std::string retval;
    thrust::copy(this->ptr, this->ptr + this->len, back_inserter(retval));
    return retval;
}


void PCudaString::destroy() {
    if (this->str) {
        thrust::device_free(this->ptr);
        this->str = NULL;
        this->len = -1;
    }
}

bool operator< (PCudaString lhs, PCudaString rhs) {
    char *l = lhs.str;
    char *r = rhs.str;
    while((*l && *r) && *l == *r) {
        ++l;
        ++r;
    }
    return *l < *r;
}

bool pcuda_integer_sort(std::vector<long> *data) {
    thrust::device_vector<long> device = *data;
    thrust::sort(device.begin(), device.end());
    thrust::copy(device.begin(), device.end(), data->begin());
    return true;
}

bool pcuda_float_sort(std::vector<double> *data) {
    thrust::device_vector<double> device = *data;
    thrust::sort(device.begin(), device.end());
    thrust::copy(device.begin(), device.end(), data->begin());
    return true;
}

bool pcuda_string_sort(std::vector<std::string> *data) {
    printf("In pcuda_string_sort\n");
    thrust::device_vector<PCudaString> device;
    printf("Reserving memory\n");
    device.reserve(data->size());
    printf("Copying data to device\n");
    for (std::vector<std::string>::iterator iter = data->begin();
         iter != data->end(); ++iter) {
        std::string s = *iter;
        device.push_back(s);
    }
    printf("On-device sort\n");
    thrust::sort(device.begin(), device.end());
    printf("Copying data from device\n");
    thrust::host_vector<PCudaString> results = device;
    data->clear();
    for (thrust::host_vector<PCudaString>::iterator iter = results.begin();
         iter != results.end(); ++iter) {
        PCudaString cs = *iter;
        std::string s = cs;
        cs.destroy();
        data->push_back(s);
    }
    printf("Done!\n");
    return true;
}

bool pcuda_integer_binary_search(std::vector<long> *data, const long target) {
    thrust::device_vector<long> device = *data;
    return thrust::binary_search(device.begin(), device.end(), target, thrust::less<long>());
}

bool pcuda_float_binary_search(std::vector<double> *data, double target) {
    thrust::device_vector<double> device = *data;
    return thrust::binary_search(device.begin(), device.end(), target);
}

void pcuda_integer_intersection(std::vector<long> *first, std::vector<long> *second,
                                std::vector<long> *intersection) {
    thrust::set_intersection(first->begin(), first->end(),
                             second->begin(), second->end(), std::back_inserter(*intersection));
}

void pcuda_float_intersection(std::vector<double> *first, std::vector<double> *second,
                                std::vector<double> *intersection) {
    thrust::set_intersection(first->begin(), first->end(),
                             second->begin(), second->end(), std::back_inserter(*intersection));
}

void pcuda_integer_minmax(std::vector<long> *data, long *minmax) {
    thrust::pair<std::vector<long>::iterator,
                 std::vector<long>::iterator> result = thrust::minmax_element(data->begin(), data->end());
    minmax[0] = *result.first;
    minmax[1] = *result.second;
}

void pcuda_float_minmax(std::vector<double> *data, double *minmax) {
    thrust::pair<std::vector<double>::iterator,
                 std::vector<double>::iterator> result = thrust::minmax_element(data->begin(), data->end());
    minmax[0] = *result.first;
    minmax[1] = *result.second;
}

struct CastToFloat
{
    float operator()(double value) const { return static_cast<float>(value);}
};

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void pcuda_mmul(std::vector<double> *a, std::vector<double> *b, std::vector<double> *c,  const int m, const int k, const int n){
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a;
    thrust::device_vector<float> d_b;
    thrust::device_vector<float> d_c;

    std::transform(a->begin(), a->end(), std::back_inserter(d_a), CastToFloat());
    std::transform(b->begin(), b->end(), std::back_inserter(d_b), CastToFloat());
    std::transform(c->begin(), c->end(), std::back_inserter(d_c), CastToFloat());

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, thrust::raw_pointer_cast(&d_a[0]), lda, thrust::raw_pointer_cast(&d_b[0]), ldb, beta, thrust::raw_pointer_cast(&d_c[0]), ldc);
    //std::cout << "\ncublasSgemm Status = " << res << std::endl;

    thrust::copy(d_c.begin(), d_c.end(), c->begin());
    // Destroy the handle
    cublasDestroy(handle);
}

void pcuda_gemv(const int m, const int n, const double alpha, std::vector<double> *a, std::vector<double> *x,const double beta, std::vector<double> *y){
    int lda=m;
    const float alf = (float)alpha;
    const float bet = (float)beta;
    const float *_alpha = &alf;
    const float *_beta =  &bet;
    int incx=1, incy=1;

    //Fallback to float to support cuda architecture < 1.3  
    thrust::device_vector<float> d_a;
    thrust::device_vector<float> d_x;
    thrust::device_vector<float> d_y;

    std::transform(a->begin(), a->end(), std::back_inserter(d_a), CastToFloat());
    std::transform(x->begin(), x->end(), std::back_inserter(d_x), CastToFloat());
    std::transform(y->begin(), y->end(), std::back_inserter(d_y), CastToFloat());
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasStatus_t res = cublasSgemv(handle, CUBLAS_OP_N, m, n, _alpha, thrust::raw_pointer_cast(&d_a[0]), lda, thrust::raw_pointer_cast(&d_x[0]), incx, _beta, thrust::raw_pointer_cast(&d_y[0]), incy);
    //std::cout << "\ncublasSgemv Status = " << res << std::endl;

    thrust::copy(d_y.begin(), d_y.end(), y->begin());

    // Destroy the handle
    cublasDestroy(handle);
}

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

//SAXPY:  y <- a * x + y
void pcuda_saxpy(double a, std::vector<double> *x, std::vector<double> *y)
{
    
    const float _a = (float)a;
    thrust::device_vector<float> d_x;
    thrust::device_vector<float> d_y;

    std::transform(x->begin(), x->end(), std::back_inserter(d_x), CastToFloat());
    std::transform(y->begin(), y->end(), std::back_inserter(d_y), CastToFloat());

    thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), saxpy_functor(_a));

    thrust::copy(d_y.begin(), d_y.end(), y->begin());

}