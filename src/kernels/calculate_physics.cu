#include "calculate_physics.h"
#include "physics_constants.h"

// calculate the heat capacity from kappa and meanmolmass
// TODO: understand when this is needed
/*
__global__ void calculate_cp(
			     double* kappa,
			     double* meanmolmass_lay,
			     double* c_p_lay,
			     int nlayer)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){
        
        c_p_lay[i] = KBOLTZMANN / (kappa[i] * meanmolmass_lay[i]);
    }
}
*/


