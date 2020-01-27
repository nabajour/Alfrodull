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



//  calculates the transmission function
__device__ double trans_func(
        double epsi, 
        double delta_tau, 
        double w0, 
        double g0
){

    return exp(-1.0/epsi*sqrt((1.0-w0*g0)*(1.0-w0))*delta_tau);
}


// calculates the G+ function
__device__ double G_plus_func(
        double w0, 
        double g0, 
        double epsi, 
        double mu_star){

    double num = (1.0 - w0) * (1.0 - w0 * g0) - 1.0;

    double denom = pow(mu_star,-2.0) - pow(epsi,-2.0) * (1.0 - w0) * (1.0 - w0 * g0);

    double second_term = 1.0/epsi + 1.0/(mu_star * (1.0 - w0 * g0));
    
    double third_term = w0 * g0 * mu_star / (1.0 - w0 * g0);
            
    double bracket = num/denom * second_term + third_term;

    double result =  0.5 * bracket;

    return result;
}


// calculates the G- function
__device__ double G_minus_func(
        double w0, 
        double g0, 
        double epsi, 
        double mu_star){

    double num = (1.0 - w0) * (1.0 - w0 * g0) - 1.0;

    double denom = pow(mu_star,-2.0) - pow(epsi,-2.0) * (1.0 - w0) * (1.0 - w0 * g0);

    double second_term = 1.0/epsi - 1.0/(mu_star * (1.0 - w0 * g0));
    
    double third_term = w0 * g0 * mu_star / (1.0 - w0 * g0);
            
    double bracket = num/denom * second_term - third_term;

    double result =  0.5 * bracket;

    return result;
}


// limiting the values of the G_plus and G_minus coefficients to less than 1e8. 
// This value is somewhat ad hoc from visual analysis. To justify, results are quite insensitive to this value.
__device__ double G_limiter(double G){
    
    if(abs(G) < 1e8){
        return G;	
    }
    else{
        return 0;
    }
}



// calculates the single scattering albedo w0
__device__ double single_scat_alb(
        double scat_cross, 
        double opac_abs, 
        double meanmolmass
){

    return scat_cross / (scat_cross + opac_abs*meanmolmass);
}



// fitting function for the E parameter according to "Heng, Malik & Kitzmann 2018
__device__ double E_parameter(
        double w0, 
        double g0
){
    double E;
    
    if (w0 > 0 && g0 >= 0){
        
        E = max(1.0, 1.225 - 0.1582*g0 - 0.1777*w0 - 0.07465*pow(1.0*g0, 2.0) + 0.2351*w0*g0 - 0.05582*pow(w0, 2.0));
    }
    else{
        E = 1.0;
    }
    return E;
}


// calculates the two-stream coupling coefficient Zeta_minus with the scattering coefficient E
__device__ double zeta_minus(
        double w0, 
        double g0,
        int scat_corr
){
    double E;

    if(scat_corr==1){
        E = E_parameter(w0, g0);
    }
    else{
        E = 1.0;
    }
    return 0.5 * (1.0 - sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}


// calculates the two-stream coupling coefficient Zeta_plus with the scattering coefficient E
__device__ double zeta_plus(
        double w0, 
        double g0,
        int scat_corr
){
    double E;

    if(scat_corr==1){
        E = E_parameter(w0, g0);
    }
    else{
        E = 1.0;
    }
    return 0.5 * (1.0 + sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}





// calculation of transmission, w0, zeta-functions, and capital letters for the layer centers in the isothermal case
// TODO: check ny meaning
// kernel runs per wavelength bin, per wavelength sampling (?) and per layer
__global__ void trans_iso(
        double* 	trans_wg,
        double* 	delta_tau_wg,
        double* 	M_term,
        double* 	N_term,
        double* 	P_term,
        double* 	G_plus,
        double* 	G_minus,
        double* 	delta_colmass,
        double* 	opac_wg_lay,
        double* cloud_opac_lay,
        double* 	meanmolmass_lay,
        double* 	scat_cross_lay,
        double* 	cloud_scat_cross_lay,
        double*  w_0,
        double* 	g_0_tot_lay,
        double   g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
){
    // indices
    // wavelength bin
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // sampling point (?)
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // layer
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer) {

        double ray_cross;
        double cloud_cross;
        double g0 = g_0;

        if(clouds == 1){
            g0 = g_0_tot_lay[x + nbin*i];
        }

        if (scat == 1){
            ray_cross = scat_cross_lay[x + nbin*i];
            cloud_cross = cloud_scat_cross_lay[x + nbin*i];
        }
        else{
            ray_cross = 0;
            cloud_cross = 0;
        }

        w_0[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross + cloud_cross, opac_wg_lay[y+ny*x + ny*nbin*i] + cloud_opac_lay[i], meanmolmass_lay[i]);
        double w0 = w_0[y+ny*x + ny*nbin*i];

        delta_tau_wg[y+ny*x + ny*nbin*i] = delta_colmass[i] * (opac_wg_lay[y+ny*x + ny*nbin*i] + cloud_opac_lay[i] + (ray_cross + cloud_cross)/meanmolmass_lay[i]);
        double del_tau = delta_tau_wg[y+ny*x + ny*nbin*i];
        trans_wg[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau, w0, g0);
        double trans = trans_wg[y+ny*x + ny*nbin*i];

        double zeta_min = zeta_minus(w0, g0, scat_corr);
        double zeta_pl = zeta_plus(w0, g0, scat_corr);

        M_term[y+ny*x + ny*nbin*i] = (zeta_min*zeta_min) * (trans*trans) - (zeta_pl*zeta_pl);
        N_term[y+ny*x + ny*nbin*i] = zeta_pl * zeta_min * (1.0 - (trans*trans));
        P_term[y+ny*x + ny*nbin*i] = ((zeta_min*zeta_min) - (zeta_pl*zeta_pl)) * trans;
                
        G_plus[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w0, g0, epsi, mu_star));
        G_minus[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w0, g0, epsi, mu_star));
    }
}

// calculation of transmission, w0, zeta-functions, and capital letters for the non-isothermal case
__global__ void trans_noniso(
        double* trans_wg_upper,
        double* trans_wg_lower,
        double* delta_tau_wg_upper,
        double* delta_tau_wg_lower,
        double* M_upper,
        double* M_lower,
        double* N_upper,
        double* N_lower,
        double* P_upper,
        double* P_lower,
        double* G_plus_upper,
        double* G_plus_lower,
        double* G_minus_upper,
        double* G_minus_lower,
        double* delta_col_upper,
        double* delta_col_lower,
        double* opac_wg_lay,
        double* opac_wg_int,
        double* cloud_opac_lay,
        double* cloud_opac_int,		
        double* meanmolmass_lay,
        double* meanmolmass_int,
        double* scat_cross_lay,
        double* scat_cross_int,
        double* cloud_scat_cross_lay,
        double* cloud_scat_cross_int,		
        double* w_0_upper,
        double* w_0_lower,
        double* 	g_0_tot_lay,
        double* 	g_0_tot_int,
        double	g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer){

        double ray_cross_up;
        double ray_cross_low;
        double cloud_cross_up;
        double cloud_cross_low;
        double g0_up = g_0;
        double g0_low = g_0;
        
        if(clouds == 1){
            g0_up = (g_0_tot_lay[x + nbin*i] + g_0_tot_int[x + nbin*(i+1)]) / 2.0;
            g0_low = (g_0_tot_int[x + nbin*i] + g_0_tot_lay[x + nbin*i]) / 2.0;
        }

        if (scat == 1){
            ray_cross_up = (scat_cross_lay[x + nbin*i] + scat_cross_int[x + nbin*(i+1)]) / 2.0;
            ray_cross_low = (scat_cross_int[x + nbin*i] + scat_cross_lay[x + nbin*i]) / 2.0;
            cloud_cross_up = (cloud_scat_cross_lay[x + nbin*i] + cloud_scat_cross_int[x + nbin*(i+1)]) / 2.0;
            cloud_cross_low = (cloud_scat_cross_int[x + nbin*i] + cloud_scat_cross_lay[x + nbin*i]) / 2.0;
        }
        else{
            ray_cross_up = 0;
            ray_cross_low = 0;
            cloud_cross_up = 0;
            cloud_cross_low = 0;
        }
        
        double opac_up = (opac_wg_lay[y+ny*x + ny*nbin*i]+opac_wg_int[y+ny*x + ny*nbin*(i+1)]) / 2.0;
        double opac_low = (opac_wg_int[y+ny*x + ny*nbin*i]+opac_wg_lay[y+ny*x + ny*nbin*i]) / 2.0;
        double cloud_opac_up = (cloud_opac_lay[i] + cloud_opac_int[i+1]) / 2.0;
        double cloud_opac_low = (cloud_opac_int[i] + cloud_opac_lay[i]) / 2.0;

        double meanmolmass_up = (meanmolmass_lay[i] + meanmolmass_int[i+1]) / 2.0;
        double meanmolmass_low = (meanmolmass_int[i] + meanmolmass_lay[i]) / 2.0;
        
        w_0_upper[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross_up + cloud_cross_up, opac_up + cloud_opac_up, meanmolmass_up);
        double w_0_up = w_0_upper[y+ny*x + ny*nbin*i];
        w_0_lower[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross_low + cloud_cross_low, opac_low + cloud_opac_low, meanmolmass_low);
        double w_0_low = w_0_lower[y+ny*x + ny*nbin*i];

        delta_tau_wg_upper[y+ny*x + ny*nbin*i] = delta_col_upper[i] * (opac_up + cloud_opac_up + (ray_cross_up + cloud_cross_up)/meanmolmass_up);
        double del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i];
        delta_tau_wg_lower[y+ny*x + ny*nbin*i] = delta_col_lower[i] * (opac_low + cloud_opac_low + (ray_cross_low + cloud_cross_low)/meanmolmass_low);
        double del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i];

        trans_wg_upper[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_up, w_0_up, g0_up);
        double trans_up = trans_wg_upper[y+ny*x + ny*nbin*i];
        trans_wg_lower[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_low, w_0_low, g0_low);
        double trans_low = trans_wg_lower[y+ny*x + ny*nbin*i];
        
        double zeta_min_up = zeta_minus(w_0_up, g0_up, scat_corr);
        double zeta_min_low = zeta_minus(w_0_low, g0_low, scat_corr);
        double zeta_pl_up = zeta_plus(w_0_up, g0_up, scat_corr);		
        double zeta_pl_low = zeta_plus(w_0_low, g0_low, scat_corr);

        M_upper[y+ny*x + ny*nbin*i] = (zeta_min_up*zeta_min_up) * (trans_up*trans_up) - (zeta_pl_up*zeta_pl_up);
        M_lower[y+ny*x + ny*nbin*i] = (zeta_min_low*zeta_min_low) * (trans_low*trans_low) - (zeta_pl_low*zeta_pl_low);
        N_upper[y+ny*x + ny*nbin*i] = zeta_pl_up * zeta_min_up * (1.0 - (trans_up*trans_up));
        N_lower[y+ny*x + ny*nbin*i] = zeta_pl_low * zeta_min_low * (1.0 - (trans_low*trans_low));
        P_upper[y+ny*x + ny*nbin*i] = ((zeta_min_up*zeta_min_up) - (zeta_pl_up*zeta_pl_up)) * trans_up;
        P_lower[y+ny*x + ny*nbin*i] = ((zeta_min_low*zeta_min_low) - (zeta_pl_low*zeta_pl_low)) * trans_low;

        G_plus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_up, g0_up, epsi, mu_star));
        G_plus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_low, g0_low, epsi, mu_star));
        G_minus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_up, g0_up, epsi, mu_star));
        G_minus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_low, g0_low, epsi, mu_star));

    }
}
