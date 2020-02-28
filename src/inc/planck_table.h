#pragma once

#include "cuda_device_memory.h"

// adjust the incident flux to correspond to the correct brightness temperature
__global__ void corr_inc_energy(double* planck_grid,
                                double* starflux,
                                double* deltalambda,
                                bool    realstar,
                                int     nwave,
                                double  Tstar,
                                int     dim);

class planck_table
{
public:
    planck_table();

    ~planck_table();


    void construct_planck_table(double* lambda_edge, double* deltalambda, int nwave, double Tstar);

    cuda_device_memory<double> planck_grid;

    int    dim   = 0;
    int    step  = 0;
    double Tstar = 0;
    int    nplanck_grid;
};
