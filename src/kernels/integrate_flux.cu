
#include "atomic_add.h"
#include "physics_constants.h"

// calculates the integrated upwards and downwards fluxes
__global__ void integrate_flux_double(
        double* deltalambda, 
        double* F_down_tot, 
        double* F_up_tot, 
        double* F_net, 
        double* F_down_wg, 
        double* F_up_wg,
        double* F_dir_wg,
        double* F_down_band, 
        double* F_up_band, 
        double* F_dir_band,
        double* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            F_up_tot[i] = 0;
            F_down_tot[i] = 0;

            F_dir_band[x + nbin * i] = 0;
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            while(x < nbin && y < ny && i < numinterfaces){
                
                atomicAdd_double(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;
    }
    __syncthreads();
    
    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){
        
        while(x < nbin && y == 0 && i < numinterfaces){

            atomicAdd_double(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
            atomicAdd_double(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < 1 && i < numinterfaces){
        F_net[i] = F_up_tot[i] - F_down_tot[i];
        i += blockDim.z;
    }
}

