#pragma once

#include <memory>

__device__ double analyt_planck(int n, double y1, double y2);


__device__ double power_int(double x, int i);

__host__ __device__ void thomas_solve(double4* A,
                                      double4* B,
                                      double4* C,
                                      double2* D,
                                      double4* C_prime,
                                      double2* D_prime,
                                      double2* X,
                                      int      N);

//***************************************************************************************************
__global__ void arrays_mean(double* array1, double* array2, double* array_out, int array_size);


//***************************************************************************************************

// first simple integration over weights
__global__ void integrate_val_band(double* val_wg,       // in
                                   double* val_band,     // out
                                   double* gauss_weight, // in
                                   int     nbin,
                                   int     num_val,
                                   int     ny);

// simple integration over bins/bands
__global__ void integrate_val_tot(double* val_tot,     // out
                                  double* val_band,    // in
                                  double* deltalambda, // in
                                  int     nbin,
                                  int     num_val);

std::shared_ptr<double[]>
integrate_band(double* val, double* gauss_weight, int num_val, int nbin, int ny);

std::shared_ptr<double[]> integrate_wg_band(double* val,
                                            double* gauss_weight,
                                            double* deltalambda,
                                            int     num_val,
                                            int     nbin,
                                            int     ny);
