#include "correct_surface_emission.h"
#include "physics_constants.h"


// adjust the surface emission to satisfy the Stefan-Boltzmann law for the surface temperature
__global__ void correct_surface_emission(
				      double*  F_down_tot,
				      double*  delta_lambda,
				      double* 	planckband_lay,
				      double   surf_albedo,
				      double   T_surf,
				      int     nwave,
				      int     numlayers,
				      int     itervalue
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (x < nwave){
        
        utype uncorr_emission = 0;
        
        for (int xl = 0; xl < nwave; xl++){
            
            uncorr_emission += delta_lambda[xl] * PI * planckband_lay[(numlayers+1) + xl * (numlayers + 2)];
        }
        
        utype corr_factor = 1.0;
        
        if(uncorr_emission > 0){
            corr_factor = STEFANBOLTZMANN * pow(T_surf, 4.0) / uncorr_emission;
        }
        // correction info -- for debugging purposes
        //if(x == 0 and itervalue % 100 == 0) printf("Surface emission corrected by %.2f ppm.\n", 1e6 * (corr_factor - 1.0));
        
        planckband_lay[(numlayers+1) + x * (numlayers + 2)] *= corr_factor;
    }
}
