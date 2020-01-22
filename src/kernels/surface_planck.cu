#include "surface_planck.h"

#include "physics_constants.h"


// calculates the planck function for given surface temperature
__global__ void calc_surface_planck(
    double* planckband_lay, 
    double* lambda_edge, 
    double* deltalambda,
    int 	nwave,
    int     numlayers,
    double 	T_surf
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (x < nwave) {
        
      double shifty;
      double D;
      double y_bot;
      double y_top;

      planckband_lay[numlayers+1 + x * (numlayers + 2)] = 0.0;

      // analytical calculation, only for T_surf > 0
      if(T_surf > 0.01){
	D = 2.0 * (power_int(KBOLTZMANN / HCONST, 3) * KBOLTZMANN * power_int(T_surf, 4)) / (CSPEED*CSPEED);
	y_top = HCONST * CSPEED / (lambda_edge[x+1] * KBOLTZMANN * T_surf);
	y_bot = HCONST * CSPEED / (lambda_edge[x] * KBOLTZMANN * T_surf);
	
	// rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
	if(y_bot < y_top){
	  shifty = y_top;
	  y_top = y_bot;
	  y_bot = shifty;
	}
	
            for(int n=1;n<200;n++){
	      planckband_lay[numlayers+1 + x * (numlayers + 2)] += D * analyt_planck(n, y_bot, y_top);
            }
      }
      planckband_lay[numlayers+1 + x * (numlayers + 2)] /= deltalambda[x];
    }
}
