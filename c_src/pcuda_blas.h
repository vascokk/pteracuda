#ifndef PCUDA_BLAS
#define PCUDA_BLAS

#include <vector>

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

void pcuda_gemm(const int transpose_a, const int transpose_b, const int m, const int n, const int k, const double alpha, std::vector<double> *a, std::vector<double> *b, const double beta, std::vector<double> *c);
void pcuda_gemv(const int transpose, const int m, const int n, const double alpha, std::vector<double> *a, std::vector<double> *x,const double beta, std::vector<double> *y);
void pcuda_saxpy(double a, std::vector<double> *x, std::vector<double> *y);
void pcuda_transpose(const int m, const int n, std::vector<double> *a, std::vector<double> *b);
void pcuda_geam(const int transpose_a, const int transpose_b, const int m, const int n, const double alpha, std::vector<double> *a, const double beta, std::vector<double> *b, std::vector<double> *c);
void pcuda_smm(const double alpha, std::vector<double> *a, std::vector<double> *b);
  
#endif