#include "cudaDeviceMemory.h"


class planck_table
{
public:
  planck_table();

  ~planck_table();

  construct_planck_table(        double* lambda_edge, 
				 double* deltalambda,
				 int 	nwave, 
				 double 	Tstar, 
				 int     dim,
				 int     step);

  cuda_device_memory<double> planck_grid;

  int dim = 0;
  int step = 0;
  double Tstar = 0;
  int nplanck_grid;
};
