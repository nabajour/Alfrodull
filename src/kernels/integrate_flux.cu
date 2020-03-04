
#include "atomic_add.h"
#include "calculate_physics.h"
#include "physics_constants.h"

#include <stdio.h>

// calculates the integrated upwards and downwards fluxes
/*
  NOTE: called as this:

    dim3 threadsPerBlock(1, 1, 1); <- grid dim, defines BlockIdx
    dim3 numBlocks(32, 4, 8);      <- threads in a block, -> defines blockDim and threadIdx

    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    //printf("Running Alfrodull Wrapper for integrate flux\n");
    integrate_flux_double<<<threadsPerBlock, numBlocks>>>(deltalambda,
*/
__global__ void integrate_flux_double(double* deltalambda,  // in
                                      double* F_down_tot,   // out
                                      double* F_up_tot,     // out
                                      double* F_net,        // out
                                      double* F_down_wg,    // in
                                      double* F_up_wg,      // in
                                      double* F_dir_wg,     // in
                                      double* F_down_band,  // out
                                      double* F_up_band,    // out
                                      double* F_dir_band,   // out
                                      double* gauss_weight, // in
                                      int     nbin,
                                      int     numinterfaces,
                                      int     ny) {

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    // set memory to 0.
    
    if (y == 0) {
        while (i < numinterfaces) {
            while (x < nbin) {

                F_up_tot[i]   = 0;
                F_down_tot[i] = 0;

                F_dir_band[x + nbin * i]  = 0;
                F_up_band[x + nbin * i]   = 0;
                F_down_band[x + nbin * i] = 0;

                x += blockDim.x;
            }
            x = threadIdx.x;
            i += blockDim.z;
        }
    }
    __syncthreads();

    i = threadIdx.z;

    while (i < numinterfaces) {
        while (y < ny) {
            while (x < nbin) {

                atomicAdd_double(&(F_dir_band[x + nbin * i]),
                                 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_up_band[x + nbin * i]),
                                 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_down_band[x + nbin * i]),
                                 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);

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

    if (y == 0) {
        while (i < numinterfaces) {
            while (x < nbin) {

                atomicAdd_double(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
                atomicAdd_double(&(F_down_tot[i]),
                                 (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i])
                                     * deltalambda[x]);

                x += blockDim.x;
            }
            x = threadIdx.x;
            i += blockDim.z;
        }
    }
    __syncthreads();

    i = threadIdx.z;

    if (x == 0 && y == 0) {
        while (i < numinterfaces) {
            F_net[i] = F_up_tot[i] - F_down_tot[i];
            i += blockDim.z;
        }
    }
}

/* 
numinterfaces: levels
nbin:          frequency bins
ny:            weights in frequency bins

must sum:
* over weighted ny into frequency bins (down_band, up_band. dir_band)
* over frequency bins into total flux per levels (up_tot, down_tot, net)
*/
// first simple integration over weights
__global__ void integrate_flux_band(double* F_down_wg,    // in
				    double* F_up_wg,      // in
				    double* F_dir_wg,     // in
				    double* F_down_band,  // out
				    double* F_up_band,    // out
				    double* F_dir_band,   // out
				    double* gauss_weight, // in
				    int     nbin,
				    int     numinterfaces,
				    int     ny) {

  int interface_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;
  

  if (interface_idx < numinterfaces && bin_idx < nbin) {
    // set memory to 0.
    F_dir_band[bin_idx + nbin * interface_idx]  = 0;
    F_up_band[bin_idx + nbin * interface_idx]   = 0;
    F_down_band[bin_idx + nbin * interface_idx] = 0;
    
    int bin_offset = bin_idx + nbin * interface_idx;
    
    for (int y = 0; y < ny; y++) {
      double w = gauss_weight[y];
      int weight_offset = y + ny * bin_idx + ny * nbin * interface_idx;
      
      F_dir_band[bin_offset] += 0.5 * w * F_dir_wg[weight_offset];
      F_up_band[bin_offset] += 0.5 * w * F_up_wg[weight_offset];
      F_down_band[bin_offset] += 0.5 * w * F_down_wg[weight_offset];
    }
  }
}

// simple integration over bins/bands
__global__ void integrate_flux_tot(double* deltalambda,  // in
				   double* F_down_tot,   // out
				   double* F_up_tot,     // out
				   double* F_net,        // out
				   double* F_down_band,  // out
				   double* F_up_band,    // out
				   double* F_dir_band,   // out
				   int     nbin,
				   int     numinterfaces) {

  
  int interface_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (interface_idx < numinterfaces) {
    
    F_up_tot[interface_idx]   = 0;
    F_down_tot[interface_idx] = 0;

    for (int bin = 0; bin < nbin; bin++) {
      int band_idx = interface_idx*nbin + bin;
      F_up_tot[interface_idx] += F_up_band[band_idx] * deltalambda[bin];
      F_down_tot[interface_idx] += F_down_band[band_idx] * deltalambda[bin] + F_dir_band[band_idx];
    }
    
    __syncthreads();
    F_net[interface_idx] = F_up_tot[interface_idx] - F_down_tot[interface_idx];
  }
}


// calculates the direct beam flux with geometric zenith angle correction, isothermal version
__global__ void fdir_iso(double* F_dir_wg,       // out
                         double* planckband_lay, // in
                         double* delta_tau_wg,   // in
                         double* z_lay,          // in
                         double  mu_star,
                         double  R_planet,
                         double  R_star,
                         double  a,
                         bool    dir_beam,
                         bool    geom_zenith_corr,
                         int     ninterface,
                         int     nbin,
                         int     ny) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        double I_dir = ((R_star / a) * (R_star / a)) * PI
                       * planckband_lay[(ninterface - 1) + x * (ninterface - 1 + 2)];

        // initialize each flux value
	if (dir_beam)
	  F_dir_wg[y + ny * x + ny * nbin * i] = -mu_star * I_dir;
	else
	  F_dir_wg[y + ny * x + ny * nbin * i] = 0.0;
        double mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for (int j = ninterface - 2; j >= i; j--) {

            if (geom_zenith_corr) {
                mu_star_layer_j = -sqrt(1.0
                                        - pow((R_planet + z_lay[i]) / (R_planet + z_lay[j]), 2.0)
                                              * (1.0 - pow(mu_star, 2.0)));
            }
            else {
                mu_star_layer_j = mu_star;
            }

            // direct stellar flux
            F_dir_wg[y + ny * x + ny * nbin * i] *=
                exp(delta_tau_wg[y + ny * x + ny * nbin * j] / mu_star_layer_j);
        }
    }
}


// calculates the direct beam flux with geometric zenith angle correction, non-isothermal version
__global__ void fdir_noniso(double* F_dir_wg,
                            double* Fc_dir_wg,
                            double* planckband_lay,
                            double* delta_tau_wg_upper,
                            double* delta_tau_wg_lower,
                            double* z_lay,
                            double  mu_star,
                            double  R_planet,
                            double  R_star,
                            double  a,
                            bool    dir_beam,
                            bool    geom_zenith_corr,
                            int     ninterface,
                            int     nbin,
                            int     ny) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        double I_dir = ((R_star / a) * (R_star / a)) * PI
                       * planckband_lay[(ninterface - 1) + x * (ninterface - 1 + 2)];

        // initialize each flux value
	if (dir_beam)
	  F_dir_wg[y + ny * x + ny * nbin * i] = -mu_star * I_dir;
	else
	  F_dir_wg[y + ny * x + ny * nbin * i] = 0.0;

        double mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for (int j = ninterface - 2; j >= i; j--) {

            if (geom_zenith_corr) {
                mu_star_layer_j = -sqrt(1.0
                                        - pow((R_planet + z_lay[i]) / (R_planet + z_lay[j]), 2.0)
                                              * (1.0 - pow(mu_star, 2.0)));
            }
            else {
                mu_star_layer_j = mu_star;
            }

            double delta_tau = delta_tau_wg_upper[y + ny * x + ny * nbin * j]
                               + delta_tau_wg_lower[y + ny * x + ny * nbin * j];

            // direct stellar flux
            Fc_dir_wg[y + ny * x + ny * nbin * i] =
                F_dir_wg[y + ny * x + ny * nbin * i]
                * exp(delta_tau_wg_upper[y + ny * x + ny * nbin * j] / mu_star_layer_j);
            F_dir_wg[y + ny * x + ny * nbin * i] *= exp(delta_tau / mu_star_layer_j);
        }
    }
}


// calculation of the spectral fluxes, isothermal case with emphasis on on-the-fly calculations
__global__ void fband_iso_notabu(double* F_down_wg,      // out
                                 double* F_up_wg,        // out
                                 double* F_dir_wg,       // in
                                 double* planckband_lay, // in
                                 double* w_0,            // in
                                 double* M_term,         // in
                                 double* N_term,         // in
                                 double* P_term,         // in
                                 double* G_plus,         // in
                                 double* G_minus,        // in
                                 double* g_0_tot_lay,    // in (clouds)
                                 double  g_0,
                                 bool    singlewalk,
                                 double  Rstar,
                                 double  a,
                                 int     numinterfaces,
                                 int     nbin,
                                 double  f_factor,
                                 double  mu_star,
                                 int     ny,
                                 double  epsi,
                                 bool    dir_beam,
                                 bool    clouds,
                                 bool    scat_corr,
                                 double  albedo,
                                 bool    debug,
                                 double  i2s_transition) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < nbin && y < ny) {

        double w0;
        double M;
        double N;
        double P;
        double G_pl;
        double G_min;
        double g0;

        double E;

        double flux_terms;
        double planck_terms;
        double direct_terms;

        // calculation of downward fluxes from TOA to BOA
        for (int i = numinterfaces - 1; i >= 0; i--) {

            // TOA boundary -- incoming stellar flux
            if (i == numinterfaces - 1) {
	      if (dir_beam)		
		F_down_wg[y + ny * x + ny * nbin * i] = 0.0;
	      else
		F_down_wg[y + ny * x + ny * nbin * i] =
		  f_factor * ((Rstar / a) * (Rstar / a)) * PI
		* planckband_lay[i + x * (numinterfaces - 1 + 2)];

            }
            else {
                w0    = w_0[y + ny * x + ny * nbin * i];
                M     = M_term[y + ny * x + ny * nbin * i];
                N     = N_term[y + ny * x + ny * nbin * i];
                P     = P_term[y + ny * x + ny * nbin * i];
                G_pl  = G_plus[y + ny * x + ny * nbin * i];
                G_min = G_minus[y + ny * x + ny * nbin * i];
                g0    = g_0;

                // improved scattering correction factor E
                E = 1.0;
                if (scat_corr) {
                    E = E_parameter(w0, g0, i2s_transition);
                }

                // experimental clouds functionality
                if (clouds) {
                    g0 = g_0_tot_lay[x + nbin * i];
                }

                // isothermal solution
                flux_terms = P * F_down_wg[y + ny * x + ny * nbin * (i + 1)]
                             - N * F_up_wg[y + ny * x + ny * nbin * i];

                planck_terms = planckband_lay[i + x * (numinterfaces - 1 + 2)] * (N + M - P);

                direct_terms =
                    F_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star) * (G_min * M + G_pl * N)
                    - F_dir_wg[y + ny * x + ny * nbin * (i + 1)] / (-mu_star) * P * G_min;

                direct_terms = min(0.0, direct_terms);

                F_down_wg[y + ny * x + ny * nbin * i] =
                    1.0 / M
                    * (flux_terms + 2.0 * PI * epsi * (1.0 - w0) / (E - w0) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (F_down_wg[y + ny * x + ny * nbin * i] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i,
                               x,
                               y);
                }
            }
        }

        // calculation of upward fluxes from BOA to TOA
        for (int i = 0; i < numinterfaces; i++) {

            // BOA boundary -- surface emission and reflection
            if (i == 0) {

                double reflected_part = albedo
                                        * (F_dir_wg[y + ny * x + ny * nbin * i]
                                           + F_down_wg[y + ny * x + ny * nbin * i]);

                // this is the surface/BOA emission. it correctly considers the emissivity e = (1 - albedo)
                double BOA_part =
                    (1.0 - albedo) * PI * (1.0 - w0) / (E - w0)
                    * planckband_lay[numinterfaces
                                     + x
                                           * (numinterfaces - 1
                                              + 2)]; // remember: numinterfaces = numlayers + 1

                F_up_wg[y + ny * x + ny * nbin * i] =
                    reflected_part
                    + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
            }
            else {
                w0    = w_0[y + ny * x + ny * nbin * (i - 1)];
                M     = M_term[y + ny * x + ny * nbin * (i - 1)];
                N     = N_term[y + ny * x + ny * nbin * (i - 1)];
                P     = P_term[y + ny * x + ny * nbin * (i - 1)];
                G_pl  = G_plus[y + ny * x + ny * nbin * (i - 1)];
                G_min = G_minus[y + ny * x + ny * nbin * (i - 1)];
                g0    = g_0;

                // improved scattering correction factor E
                E = 1.0;
                if (scat_corr) {
                    E = E_parameter(w0, g0, i2s_transition);
                }

                // experimental clouds functionality
                if (clouds) {
                    g0 = g_0_tot_lay[x + nbin * (i - 1)];
                }

                // isothermal solution
                flux_terms = P * F_up_wg[y + ny * x + ny * nbin * (i - 1)]
                             - N * F_down_wg[y + ny * x + ny * nbin * i];

                planck_terms = planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)] * (N + M - P);

                direct_terms =
                    F_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star) * (G_min * N + G_pl * M)
                    - F_dir_wg[y + ny * x + ny * nbin * (i - 1)] / (-mu_star) * P * G_pl;

                direct_terms = min(0.0, direct_terms);

                F_up_wg[y + ny * x + ny * nbin * i] =
                    1.0 / M
                    * (flux_terms + 2.0 * PI * epsi * (1.0 - w0) / (E - w0) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (F_up_wg[y + ny * x + ny * nbin * i] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i,
                               x,
                               y);
                }
            }
        }
    }
}

// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__global__ void fband_noniso_notabu(double* F_down_wg,
                                    double* F_up_wg,
                                    double* Fc_down_wg,
                                    double* Fc_up_wg,
                                    double* F_dir_wg,
                                    double* Fc_dir_wg,
                                    double* planckband_lay,
                                    double* planckband_int,
                                    double* w_0_upper,
                                    double* w_0_lower,
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
                                    double* g_0_tot_lay,
                                    double* g_0_tot_int,
                                    double  g_0,
                                    bool    singlewalk,
                                    double  Rstar,
                                    double  a,
                                    int     numinterfaces,
                                    int     nbin,
                                    double  f_factor,
                                    double  mu_star,
                                    int     ny,
                                    double  epsi,
                                    double  delta_tau_limit,
                                    bool    dir_beam,
                                    bool    clouds,
                                    bool    scat_corr,
                                    double  albedo,
                                    bool    debug,
                                    double  i2s_transition) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && y < ny) {

        double w0_up;
        double del_tau_up;
        double M_up;
        double N_up;
        double P_up;
        double G_pl_up;
        double G_min_up;
        double g0_up;
        double E_up;

        double w0_low;
        double del_tau_low;
        double M_low;
        double N_low;
        double P_low;
        double G_pl_low;
        double G_min_low;
        double g0_low;
        double E_low;

        double flux_terms;
        double planck_terms;
        double direct_terms;

        // calculation of downward fluxes from TOA to BOA
        for (int i = numinterfaces - 1; i >= 0; i--) {

            // TOA boundary -- incoming stellar flux
            if (i == numinterfaces - 1) {
	      if (dir_beam)
		F_down_wg[y + ny * x + ny * nbin * i] = 0.0;
	      else
                F_down_wg[y + ny * x + ny * nbin * i] =
                    f_factor * ((Rstar / a) * (Rstar / a)) * PI
                    * planckband_lay[i + x * (numinterfaces - 1 + 2)];
            }
            else {
                // upper part of layer quantities
                w0_up      = w_0_upper[y + ny * x + ny * nbin * i];
                del_tau_up = delta_tau_wg_upper[y + ny * x + ny * nbin * i];
                M_up       = M_upper[y + ny * x + ny * nbin * i];
                N_up       = N_upper[y + ny * x + ny * nbin * i];
                P_up       = P_upper[y + ny * x + ny * nbin * i];
                G_pl_up    = G_plus_upper[y + ny * x + ny * nbin * i];
                G_min_up   = G_minus_upper[y + ny * x + ny * nbin * i];
                g0_up      = g_0;

                // lower part of layer quantities
                w0_low      = w_0_lower[y + ny * x + ny * nbin * i];
                del_tau_low = delta_tau_wg_lower[y + ny * x + ny * nbin * i];
                M_low       = M_lower[y + ny * x + ny * nbin * i];
                N_low       = N_lower[y + ny * x + ny * nbin * i];
                P_low       = P_lower[y + ny * x + ny * nbin * i];
                G_pl_low    = G_plus_lower[y + ny * x + ny * nbin * i];
                G_min_low   = G_minus_lower[y + ny * x + ny * nbin * i];
                g0_low      = g_0;

                // improved scattering correction factor E
                E_up  = 1.0;
                E_low = 1.0;

                // improved scattering correction disabled for the following terms -- at least for the moment
                if (scat_corr) {
                    E_up  = E_parameter(w0_up, g0_up, i2s_transition);
                    E_low = E_parameter(w0_low, g0_low, i2s_transition);
                }

                // experimental clouds functionality
                if (clouds) {
                    g0_up  = (g_0_tot_lay[x + nbin * i] + g_0_tot_int[x + nbin * (i + 1)]) / 2.0;
                    g0_low = (g_0_tot_int[x + nbin * i] + g_0_tot_lay[x + nbin * i]) / 2.0;
                }

                // upper part of layer calculations
                if (del_tau_up < delta_tau_limit) {
                    // the isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[(i + 1) + x * numinterfaces]
                                    + planckband_lay[i + x * (numinterfaces - 1 + 2)])
                                   / 2.0 * (N_up + M_up - P_up);
                }
                else {
                    // the non-isothermal solution -- standard case
                    double pgrad_up = (planckband_lay[i + x * (numinterfaces - 1 + 2)]
                                       - planckband_int[(i + 1) + x * numinterfaces])
                                      / del_tau_up;

                    planck_terms =
                        planckband_lay[i + x * (numinterfaces - 1 + 2)] * (M_up + N_up)
                        - planckband_int[(i + 1) + x * numinterfaces] * P_up
                        + epsi / (E_up * (1.0 - w0_up * g0_up)) * (P_up - M_up + N_up) * pgrad_up;
                }
                flux_terms = P_up * F_down_wg[y + ny * x + ny * nbin * (i + 1)]
                             - N_up * Fc_up_wg[y + ny * x + ny * nbin * i];

                direct_terms =
                    Fc_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star)
                        * (G_min_up * M_up + G_pl_up * N_up)
                    - F_dir_wg[y + ny * x + ny * nbin * (i + 1)] / (-mu_star) * G_min_up * P_up;

                direct_terms = min(0.0, direct_terms);

                Fc_down_wg[y + ny * x + ny * nbin * i] =
                    1.0 / M_up
                    * (flux_terms + 2.0 * PI * epsi * (1.0 - w0_up) / (E_up - w0_up) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (Fc_down_wg[y + ny * x + ny * nbin * i] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i,
                               x,
                               y);
                }

                // lower part of layer calculations
                if (del_tau_low < delta_tau_limit) {
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[i + x * numinterfaces]
                                    + planckband_lay[i + x * (numinterfaces - 1 + 2)])
                                   / 2.0 * (N_low + M_low - P_low);
                }
                else {
                    // non-isothermal solution -- standard case
                    double pgrad_low = (planckband_int[i + x * numinterfaces]
                                        - planckband_lay[i + x * (numinterfaces - 1 + 2)])
                                       / del_tau_low;

                    planck_terms = planckband_int[i + x * numinterfaces] * (M_low + N_low)
                                   - planckband_lay[i + x * (numinterfaces - 1 + 2)] * P_low
                                   + epsi / (E_low * (1.0 - w0_low * g0_low))
                                         * (P_low - M_low + N_low) * pgrad_low;
                }
                flux_terms = P_low * Fc_down_wg[y + ny * x + ny * nbin * i]
                             - N_low * F_up_wg[y + ny * x + ny * nbin * i];

                direct_terms =
                    F_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star)
                        * (G_min_low * M_low + G_pl_low * N_low)
                    - Fc_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star) * P_low * G_min_low;

                direct_terms = min(0.0, direct_terms);

                F_down_wg[y + ny * x + ny * nbin * i] =
                    1.0 / M_low
                    * (flux_terms
                       + 2.0 * PI * epsi * (1.0 - w0_low) / (E_low - w0_low) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (F_down_wg[y + ny * x + ny * nbin * i] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i,
                               x,
                               y);
                }
            }
        }

        __syncthreads();

        // calculation of upward fluxes from BOA to TOA
        for (int i = 0; i < numinterfaces; i++) {

            // BOA boundary -- surface emission and reflection
            if (i == 0) {

                double reflected_part = albedo
                                        * (F_dir_wg[y + ny * x + ny * nbin * i]
                                           + F_down_wg[y + ny * x + ny * nbin * i]);

                // this is the surface/BOA emission. it correctly includes the emissivity e = (1 - albedo)
                double BOA_part = (1.0 - albedo) * PI * (1.0 - w0_low) / (E_low - w0_low)
                                  * planckband_lay[numinterfaces + x * (numinterfaces - 1 + 2)];

                F_up_wg[y + ny * x + ny * nbin * i] =
                    reflected_part
                    + BOA_part; // internal_part consists of the internal heat flux plus the surface/BOA emission
            }
            else {
                // lower part of layer quantities
                w0_low      = w_0_lower[y + ny * x + ny * nbin * (i - 1)];
                del_tau_low = delta_tau_wg_lower[y + ny * x + ny * nbin * (i - 1)];
                M_low       = M_lower[y + ny * x + ny * nbin * (i - 1)];
                N_low       = N_lower[y + ny * x + ny * nbin * (i - 1)];
                P_low       = P_lower[y + ny * x + ny * nbin * (i - 1)];
                G_pl_low    = G_plus_lower[y + ny * x + ny * nbin * (i - 1)];
                G_min_low   = G_minus_lower[y + ny * x + ny * nbin * (i - 1)];
                g0_low      = g_0;

                // upper part of layer quantities
                w0_up      = w_0_upper[y + ny * x + ny * nbin * (i - 1)];
                del_tau_up = delta_tau_wg_upper[y + ny * x + ny * nbin * (i - 1)];
                M_up       = M_upper[y + ny * x + ny * nbin * (i - 1)];
                N_up       = N_upper[y + ny * x + ny * nbin * (i - 1)];
                P_up       = P_upper[y + ny * x + ny * nbin * (i - 1)];
                G_pl_up    = G_plus_upper[y + ny * x + ny * nbin * (i - 1)];
                G_min_up   = G_minus_upper[y + ny * x + ny * nbin * (i - 1)];
                g0_up      = g_0;

                // improved scattering correction factor E
                E_low = 1.0;
                E_up  = 1.0;

                // improved scattering correction disabled for the following terms -- at least for the moment
                if (scat_corr) {
                    E_up  = E_parameter(w0_up, g0_up, i2s_transition);
                    E_low = E_parameter(w0_low, g0_low, i2s_transition);
                }

                // experimental clouds functionanlity
                if (clouds) {
                    g0_low =
                        (g_0_tot_int[x + nbin * (i - 1)] + g_0_tot_lay[x + nbin * (i - 1)]) / 2.0;
                    g0_up = (g_0_tot_lay[x + nbin * (i - 1)] + g_0_tot_int[x + nbin * i]) / 2.0;
                }

                // lower part of layer calculations
                if (del_tau_low < delta_tau_limit) {
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = ((planckband_int[(i - 1) + x * numinterfaces]
                                     + planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)])
                                    / 2.0 * (N_low + M_low - P_low));
                }
                else {
                    // non-isothermal solution -- standard case
                    double pgrad_low = (planckband_int[(i - 1) + x * numinterfaces]
                                        - planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)])
                                       / del_tau_low;

                    planck_terms =
                        planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)] * (M_low + N_low)
                        - planckband_int[(i - 1) + x * numinterfaces] * P_low
                        + epsi / (E_low * (1.0 - w0_low * g0_low)) * pgrad_low
                              * (M_low - P_low - N_low);
                }
                flux_terms = P_low * F_up_wg[y + ny * x + ny * nbin * (i - 1)]
                             - N_low * Fc_down_wg[y + ny * x + ny * nbin * (i - 1)];

                direct_terms =
                    Fc_dir_wg[y + ny * x + ny * nbin * (i - 1)] / (-mu_star)
                        * (G_min_low * N_low + G_pl_low * M_low)
                    - F_dir_wg[y + ny * x + ny * nbin * (i - 1)] / (-mu_star) * P_low * G_pl_low;

                direct_terms = min(0.0, direct_terms);

                Fc_up_wg[y + ny * x + ny * nbin * (i - 1)] =
                    1.0 / M_low
                    * (flux_terms
                       + 2.0 * PI * epsi * (1.0 - w0_low) / (E_low - w0_low) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (Fc_up_wg[y + ny * x + ny * nbin * (i - 1)] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i - 1,
                               x,
                               y);
                }

                // upper part of layer calculations
                if (del_tau_up < delta_tau_limit) {
                    // isothermal solution -- taken if optical depth so small that numerical instabilities may occur
                    planck_terms = (planckband_int[i + x * numinterfaces]
                                    + planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)])
                                   / 2.0 * (N_up + M_up - P_up);
                }
                else {
                    // non-isothermal solution -- standard case
                    double pgrad_up = (planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)]
                                       - planckband_int[i + x * numinterfaces])
                                      / del_tau_up;

                    planck_terms =
                        planckband_int[i + x * numinterfaces] * (M_up + N_up)
                        - planckband_lay[(i - 1) + x * (numinterfaces - 1 + 2)] * P_up
                        + epsi / (E_up * (1.0 - w0_up * g0_up)) * pgrad_up * (M_up - P_up - N_up);
                }
                flux_terms = P_up * Fc_up_wg[y + ny * x + ny * nbin * (i - 1)]
                             - N_up * F_down_wg[y + ny * x + ny * nbin * i];

                direct_terms =
                    F_dir_wg[y + ny * x + ny * nbin * i] / (-mu_star)
                        * (G_min_up * N_up + G_pl_up * M_up)
                    - Fc_dir_wg[y + ny * x + ny * nbin * (i - 1)] / (-mu_star) * P_up * G_pl_up;

                direct_terms = min(0.0, direct_terms);

                F_up_wg[y + ny * x + ny * nbin * i] =
                    1.0 / M_up
                    * (flux_terms + 2.0 * PI * epsi * (1.0 - w0_up) / (E_up - w0_up) * planck_terms
                       + direct_terms);

                //feedback if flux becomes negative
                if (debug) {
                    if (F_up_wg[y + ny * x + ny * nbin * i] < 0)
                        printf("WARNING WARNING WARNING WARNING -- negative flux found at layer: "
                               "%d, w-index: %d, y-index: %d !!! \n",
                               i,
                               x,
                               y);
                }
            }
        }
    }
}
