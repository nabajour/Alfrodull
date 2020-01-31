#include "planck_table.h"


#include "physics_constants.h"

#include "math_helpers.h"



// constructing a table with Planck function values for given wavelengths and in a suitable temperature range
__global__ void plancktable(
        double* planck_grid, 
        double* lambda_edge, 
        double* deltalambda,
        int 	nwave, 
        double 	Tstar, 
        int 	p_iter,
        int     dim,
        int     step
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int t = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nwave && t < (dim/10+1)) {

        double T;
        double shifty;
        double D;
        double y_bot;
        double y_top;

        // building flexible temperature grid from '1 K' to 'dim * 2 - 1 K' at 'step K' resolution
        // and Tstar
        if(t < (dim/10)){
                T = (t + p_iter * (dim/10)) * step + 1;
        }
        if(p_iter == 9){
            if(t == dim/10){
                T = Tstar;
            }
        }

        planck_grid[x + (t + p_iter * (dim/10)) * nwave] = 0.0;

        // analytical calculation, only for T > 0
        if(T > 0.01){
            D = 2.0 * (power_int(KBOLTZMANN / HCONST, 3) * KBOLTZMANN * power_int(T, 4)) / (CSPEED*CSPEED);
            y_top = HCONST * CSPEED / (lambda_edge[x+1] * KBOLTZMANN * T);
            y_bot = HCONST * CSPEED / (lambda_edge[x] * KBOLTZMANN * T);

            // rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
            if(y_bot < y_top){
                shifty = y_top;
                y_top = y_bot;
                y_bot = shifty;
            }

            for(int n=1;n<200;n++){
                planck_grid[x + (t + p_iter * (dim/10)) * nwave] += D * analyt_planck(n, y_bot, y_top);
            }
        }
        planck_grid[x + (t + p_iter * (dim/10)) * nwave] /= deltalambda[x];
    }
}

planck_table::planck_table()
{
  int dim = 8000;
  int step = 2;
}

~planck_table::planck_table()
{

}


void planck_table::construct_planck_table(        double* lambda_edge, 
						  double* deltalambda,
						  int 	nwave, 
						  double 	Tstar_)
{
  nplanck_grid = (dim+1)*nwave;
  Tstar = Tstar_;
  
  planck_grid.allocate(nplanck_grid);

  for (int p_iter = 0; p_iter< 10; p_iter++)
    plancktable(*planck_grid,
		lambda_edge,
		deltalambda,
		nwave, 
		Tstar, 
		p_iter,
		dim,
		step);
}
