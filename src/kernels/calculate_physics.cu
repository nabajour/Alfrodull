#include "calculate_physics.h"
#include "physics_constants.h"

#include <stdio.h>

// fitting function for the E parameter according to "Heng, Malik & Kitzmann 2018
__device__ double E_parameter(double w0, double g0, double i2s_transition) {
    double E;

    if (w0 > i2s_transition && g0 >= 0) {

        E = max(1.0,
                1.225 - 0.1582 * g0 - 0.1777 * w0 - 0.07465 * pow(1.0 * g0, 2.0) + 0.2351 * w0 * g0
                    - 0.05582 * pow(w0, 2.0));
    }
    else {
        E = 1.0;
    }
    return E;
}

//  calculates the transmission function
__device__ double trans_func(double epsi, double delta_tau, double w0, double g0, double E) {

    return exp(-1.0 / epsi * sqrt(E * (1.0 - w0 * g0) * (E - w0)) * delta_tau);
}

// calculates the G+ function
__device__ double
G_plus_func(double w0, double g0, double epsilon, double epsilon2, double mu_star, double E) {

    double num = w0 * (E * (1.0 - w0 * g0) + g0 * epsilon / epsilon2);

    double denom = E * pow(mu_star / epsilon, 2.0) * (E - w0) * (1.0 - w0 * g0) - 1.0;

    double second_term = mu_star / epsilon + 1.0 / (E * (1.0 - w0 * g0));

    double third_term = epsilon * w0 * g0 / (epsilon2 * E * (1.0 - w0 * g0));

    double bracket = num / denom * second_term + third_term;

    double result = 0.5 * bracket;

    return result;
}

// calculates the G- function
// TODO: can improve by computing E outside and pass as param ?
__device__ double
G_minus_func(double w0, double g0, double epsilon, double epsilon2, double mu_star, double E) {
    double num = w0 * (E * (1.0 - w0 * g0) + g0 * epsilon / epsilon2);

    double denom = E * pow(mu_star / epsilon, 2.0) * (E - w0) * (1.0 - w0 * g0) - 1.0;

    double second_term = mu_star / epsilon - 1.0 / (E * (1.0 - w0 * g0));

    double third_term = epsilon * w0 * g0 / (epsilon2 * E * (1.0 - w0 * g0));

    double bracket = num / denom * second_term - third_term;

    double result = 0.5 * bracket;

    return result;
}

__device__ double G_pm_denom(double w0, double g0, double epsi, double mu_star, double E) {
    double denom = E * pow(mu_star / epsi, 2.0) * (E - w0) * (1.0 - w0 * g0) - 1.0;

    return denom;
}


// limiting the values of the G_plus and G_minus coefficients to 1e8.
// This value is somewhat ad hoc from visual analysis. To justify, results are quite insensitive to this value.
__device__ double G_limiter(double G, bool debug) {

    if (abs(G) < 1e8) {
        return G;
    }
    else {
        if (debug) {
            printf("WARNING: G_functions are being artificially limited!!! \n");
        }
        return 1e8 * G / abs(G);
    }
}


// calculates the single scattering albedo w0
__device__ double single_scat_alb(double gas_scat_cross,
                                  double gas_abs,
                                  double meanmolmass,
                                  double fcloud,
                                  double cloud_scat_cross,
                                  double cloud_abs,
                                  double w_0_limit) {

    return min(
        (gas_scat_cross + fcloud * cloud_scat_cross)
            / (gas_scat_cross + gas_abs * meanmolmass + fcloud * (cloud_scat_cross + cloud_abs)),
        w_0_limit);
}

// calculates the asymmetry parameter g0
__device__ double
g0_calc(double gas_scat_cross, double fcloud, double cloud_scat_cross, double g0_cloud) {
    return (fcloud * cloud_scat_cross * g0_cloud) / (gas_scat_cross + fcloud * cloud_scat_cross);
}

// calculates the two-stream coupling coefficient Zeta_minus with the scattering coefficient E
__device__ double zeta_minus(double w0, double g0, double E) {

    return 0.5 * (1.0 - sqrt((E - w0) / (E * (1.0 - w0 * g0))));
}


// calculates the two-stream coupling coefficient Zeta_plus with the scattering coefficient E
__device__ double zeta_plus(double w0, double g0, double E) {

    return 0.5 * (1.0 + sqrt((E - w0) / (E * (1.0 - w0 * g0))));
}


// calculation of transmission, w0, zeta-functions, and capital letters for the layer centers in the isothermal case
// kernel runs per wavelength bin, per wavelength sampling (?) and per layer
__global__ void trans_iso(double* trans_wg,             // out
                          double* delta_tau_wg,         // out
                          double* M_term,               // out
                          double* N_term,               // out
                          double* P_term,               // out
                          double* G_plus,               // out
                          double* G_minus,              // out
                          double* delta_colmass,        // in
                          double* opac_wg_lay,          // in
                          double* cloud_opac_lay,       // in
                          double* meanmolmass_lay,      // in
                          double* scat_cross_lay,       // in
                          double* cloud_scat_cross_lay, // in
                          double* w_0,                  // out
                          double* g0_wg,                // out
                          double* g_0_tot_lay,          // in
                          double  g_0_gas,
                          double  epsi,
                          double  epsilon2,
                          double  mu_star,
                          double  w_0_limit,
                          bool    scat,
                          int     nbin,
                          int     ny,
                          int     nlayer,
                          double  fcloud,
                          bool    clouds,
                          bool    scat_corr,
                          bool    G_pm_limiter,
                          double  G_pm_denom_limit_for_mu_star_wiggler,
                          bool*   hit_G_pm_limit,
                          bool    debug,
                          double  i2s_transition) {
    // indices
    // wavelength bin
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // sampling point (?) y coordinate?
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // layer
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer) {

        double ray_cross;
        double cloud_cross;
        double g0       = g_0_gas;
        double g0_cloud = 0.0;
        double E        = 1.0;


        if (scat) {
            ray_cross = scat_cross_lay[x + nbin * i];
            if (clouds)
                cloud_cross = cloud_scat_cross_lay[x + nbin * i];
            else
                cloud_cross = 0.0;
            // DBG: cloud_cross = 0.0;
        }
        else {
            ray_cross   = 0.0;
            cloud_cross = 0.0;
        }

        if (clouds) {
            g0_cloud = g_0_tot_lay[x + nbin * i];
        }

        g0 = g0_calc(g_0_gas, ray_cross, fcloud, cloud_cross, g0_cloud);

        double w0 = single_scat_alb(ray_cross,
                                    opac_wg_lay[y + ny * x + ny * nbin * i],
                                    meanmolmass_lay[i],
                                    fcloud,
                                    cloud_cross,
                                    cloud_opac_lay[i],
                                    w_0_limit);


        w_0[y + ny * x + ny * nbin * i]   = w0;
        g0_wg[y + ny * x + ny * nbin * i] = g0;

        if (scat_corr) {
            E = E_parameter(w0, g0, i2s_transition);
        }

        delta_tau_wg[y + ny * x + ny * nbin * i] =
            delta_colmass[i]
            * (opac_wg_lay[y + ny * x + ny * nbin * i] + fcloud * cloud_opac_lay[i]
               + (ray_cross + fcloud * cloud_cross) / meanmolmass_lay[i]);
        double del_tau                       = delta_tau_wg[y + ny * x + ny * nbin * i];
        trans_wg[y + ny * x + ny * nbin * i] = trans_func(epsi, del_tau, w0, g0, E);
        double trans                         = trans_wg[y + ny * x + ny * nbin * i];

        double zeta_min = zeta_minus(w0, g0, E);
        double zeta_pl  = zeta_plus(w0, g0, E);

        M_term[y + ny * x + ny * nbin * i] =
            (zeta_min * zeta_min) * (trans * trans) - (zeta_pl * zeta_pl);
        N_term[y + ny * x + ny * nbin * i] = zeta_pl * zeta_min * (1.0 - (trans * trans));
        P_term[y + ny * x + ny * nbin * i] = ((zeta_min * zeta_min) - (zeta_pl * zeta_pl)) * trans;

        // DBG:
        // if (!isfinite(M_term[y + ny * x + ny * nbin * i]))
        //   printf("abnormal M_term: %g, zeta_min: %g, trans: %g, zeta_pl: %g, "
        // 	 "epsi: %g, w0: %g, delta_tau: %g g0: %g, "
        // 	 "delta_colamss: %g, opac_wg_lay: %g, cloud_opac_lay: %g, ray_cross: %g, cloud_cross: %g, meanmolmass_lay: %g\n",
        // 	 M_term[y + ny * x + ny * nbin * i], zeta_min, trans, zeta_pl,
        // 	 epsi, w0, del_tau, g0,
        // 	 delta_colmass[i], opac_wg_lay[y + ny * x + ny * nbin * i], cloud_opac_lay[i], ray_cross, cloud_cross, meanmolmass_lay[i] );
        // if (!isfinite(N_term[y + ny * x + ny * nbin * i]))
        //   printf("abnormal N_term: %g, zeta_min: %g, trans: %g, zeta_pl: %g "
        // 	 "epsi: %g, w0: %g, delta_tau: %g, g0: %g\n",
        // 	 N_term[y + ny * x + ny * nbin * i], zeta_min, trans, zeta_pl,
        // 	 epsi, w0, del_tau, g0);

        epsi     = 0.5;
        epsilon2 = 2.0 / 3.0;
        //mu_star  = -0.89;
        E = 1.0;
        if (fabs(G_pm_denom(w0, g0, epsi, mu_star, E)) < G_pm_denom_limit_for_mu_star_wiggler)
            *hit_G_pm_limit = true;

        if (G_pm_limiter) {
            G_plus[y + ny * x + ny * nbin * i] =
                G_limiter(G_plus_func(w0, g0, epsi, epsilon2, mu_star, E), debug);
            G_minus[y + ny * x + ny * nbin * i] =
                G_limiter(G_minus_func(w0, g0, epsi, epsilon2, mu_star, E), debug);
        }
        else {
            G_plus[y + ny * x + ny * nbin * i]  = G_plus_func(w0, g0, epsi, epsilon2, mu_star, E);
            G_minus[y + ny * x + ny * nbin * i] = G_minus_func(w0, g0, epsi, epsilon2, mu_star, E);
        }

        // if (!isfinite(G_plus[y + ny * x + ny * nbin * i] ))
        // printf("abnormal G_p\n");
        // if (!isfinite(G_minus[y + ny * x + ny * nbin * i] ))
        // printf("abnormal G_m\n");
    }
}

// calculation of transmission, w0, zeta-functions, and capital letters for the non-isothermal case
__global__ void trans_noniso(double* trans_wg_upper,
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
                             double* g0_wg_upper,
                             double* g0_wg_lower,
                             double* g_0_tot_lay,
                             double* g_0_tot_int,
                             double  g_0_gas,
                             double  epsi,
                             double  epsilon2,
                             double  mu_star,
                             double  w_0_limit,
                             bool    scat,
                             int     nbin,
                             int     ny,
                             int     nlayer,
                             double  fcloud,
                             bool    clouds,
                             bool    scat_corr,
                             bool    G_pm_limiter,
                             double  G_pm_denom_limit_for_mu_star_wiggler,
                             bool*   hit_G_pm_limit,
                             bool    debug,
                             double  i2s_transition) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer) {

        utype  ray_cross_up;
        utype  ray_cross_low;
        utype  cloud_cross_up;
        utype  cloud_cross_low;
        utype  g0_up        = g_0_gas;
        utype  g0_low       = g_0_gas;
        utype  g0_cloud_up  = 0.0;
        utype  g0_cloud_low = 0.0;
        double E_up         = 1.0;
        double E_low        = 1.0;

        if (scat) {
            ray_cross_up =
                (scat_cross_lay[x + nbin * i] + scat_cross_int[x + nbin * (i + 1)]) / 2.0;
            ray_cross_low = (scat_cross_int[x + nbin * i] + scat_cross_lay[x + nbin * i]) / 2.0;
            if (clouds) {
                cloud_cross_up =
                    (cloud_scat_cross_lay[x + nbin * i] + cloud_scat_cross_int[x + nbin * (i + 1)])
                    / 2.0;
                cloud_cross_low =
                    (cloud_scat_cross_int[x + nbin * i] + cloud_scat_cross_lay[x + nbin * i]) / 2.0;
            }
            else {
                cloud_cross_up  = 0.0;
                cloud_cross_low = 0.0;
            }
        }
        else {
            ray_cross_up    = 0;
            ray_cross_low   = 0;
            cloud_cross_up  = 0;
            cloud_cross_low = 0;
        }


        if (clouds) {
            g0_up  = (g_0_tot_lay[x + nbin * i] + g_0_tot_int[x + nbin * (i + 1)]) / 2.0;
            g0_low = (g_0_tot_int[x + nbin * i] + g_0_tot_lay[x + nbin * i]) / 2.0;
        }

        g0_up  = g0_calc(g_0_gas, ray_cross_up, fcloud, cloud_cross_up, g0_cloud_up);
        g0_low = g0_calc(g_0_gas, ray_cross_low, fcloud, cloud_cross_low, g0_cloud_low);

        g0_wg_upper[y + ny * x + ny * nbin * i] = g0_up;
        g0_wg_lower[y + ny * x + ny * nbin * i] = g0_low;

        utype opac_up = (opac_wg_lay[y + ny * x + ny * nbin * i]
                         + opac_wg_int[y + ny * x + ny * nbin * (i + 1)])
                        / 2.0;
        utype opac_low =
            (opac_wg_int[y + ny * x + ny * nbin * i] + opac_wg_lay[y + ny * x + ny * nbin * i])
            / 2.0;
        utype cloud_opac_up  = (cloud_opac_lay[i] + cloud_opac_int[i + 1]) / 2.0;
        utype cloud_opac_low = (cloud_opac_int[i] + cloud_opac_lay[i]) / 2.0;

        utype meanmolmass_up  = (meanmolmass_lay[i] + meanmolmass_int[i + 1]) / 2.0;
        utype meanmolmass_low = (meanmolmass_int[i] + meanmolmass_lay[i]) / 2.0;

        double w_0_up = single_scat_alb(ray_cross_up,
                                        opac_up,
                                        meanmolmass_up,
                                        fcloud,
                                        cloud_cross_up,
                                        cloud_opac_up,
                                        w_0_limit);

        w_0_upper[y + ny * x + ny * nbin * i] = w_0_up;

        double w_0_low = single_scat_alb(ray_cross_low,
                                         opac_low,
                                         meanmolmass_low,
                                         fcloud,
                                         cloud_cross_low,
                                         cloud_opac_low,
                                         w_0_limit);

        w_0_lower[y + ny * x + ny * nbin * i] = w_0_low;


        if (scat_corr) {
            E_up  = E_parameter(w_0_up, g0_up, i2s_transition);
            E_low = E_parameter(w_0_low, g0_low, i2s_transition);
        }

        delta_tau_wg_upper[y + ny * x + ny * nbin * i] =
            delta_col_upper[i]
            * (opac_up + fcloud * cloud_opac_up
               + (ray_cross_up + fcloud * cloud_cross_up) / meanmolmass_up);
        utype del_tau_up = delta_tau_wg_upper[y + ny * x + ny * nbin * i];
        delta_tau_wg_lower[y + ny * x + ny * nbin * i] =
            delta_col_lower[i]
            * (opac_low + fcloud * cloud_opac_low
               + (ray_cross_low + fcloud * cloud_cross_low) / meanmolmass_low);
        utype del_tau_low = delta_tau_wg_lower[y + ny * x + ny * nbin * i];

        trans_wg_upper[y + ny * x + ny * nbin * i] =
            trans_func(epsi, del_tau_up, w_0_up, g0_up, E_up);
        utype trans_up = trans_wg_upper[y + ny * x + ny * nbin * i];
        trans_wg_lower[y + ny * x + ny * nbin * i] =
            trans_func(epsi, del_tau_low, w_0_low, g0_low, E_low);
        utype trans_low = trans_wg_lower[y + ny * x + ny * nbin * i];

        utype zeta_min_up  = zeta_minus(w_0_up, g0_up, E_up);
        utype zeta_min_low = zeta_minus(w_0_low, g0_low, E_low);
        utype zeta_pl_up   = zeta_plus(w_0_up, g0_up, E_up);
        utype zeta_pl_low  = zeta_plus(w_0_low, g0_low, E_low);

        M_upper[y + ny * x + ny * nbin * i] =
            (zeta_min_up * zeta_min_up) * (trans_up * trans_up) - (zeta_pl_up * zeta_pl_up);
        M_lower[y + ny * x + ny * nbin * i] =
            (zeta_min_low * zeta_min_low) * (trans_low * trans_low) - (zeta_pl_low * zeta_pl_low);
        N_upper[y + ny * x + ny * nbin * i] =
            zeta_pl_up * zeta_min_up * (1.0 - (trans_up * trans_up));
        N_lower[y + ny * x + ny * nbin * i] =
            zeta_pl_low * zeta_min_low * (1.0 - (trans_low * trans_low));
        P_upper[y + ny * x + ny * nbin * i] =
            ((zeta_min_up * zeta_min_up) - (zeta_pl_up * zeta_pl_up)) * trans_up;
        P_lower[y + ny * x + ny * nbin * i] =
            ((zeta_min_low * zeta_min_low) - (zeta_pl_low * zeta_pl_low)) * trans_low;

        if ((fabs(G_pm_denom(w_0_up, g0_up, epsi, mu_star, E_up))
             < G_pm_denom_limit_for_mu_star_wiggler)
            || (fabs(G_pm_denom(w_0_low, g0_low, epsi, mu_star, E_low))
                < G_pm_denom_limit_for_mu_star_wiggler))
            *hit_G_pm_limit = true;

        if (G_pm_limiter) {
            G_plus_upper[y + ny * x + ny * nbin * i] =
                G_limiter(G_plus_func(w_0_up, g0_up, epsi, epsilon2, mu_star, E_up), debug);
            G_plus_lower[y + ny * x + ny * nbin * i] =
                G_limiter(G_plus_func(w_0_low, g0_low, epsi, epsilon2, mu_star, E_low), debug);
            G_minus_upper[y + ny * x + ny * nbin * i] =
                G_limiter(G_minus_func(w_0_up, g0_up, epsi, epsilon2, mu_star, E_up), debug);
            G_minus_lower[y + ny * x + ny * nbin * i] =
                G_limiter(G_minus_func(w_0_low, g0_low, epsi, epsilon2, mu_star, E_low), debug);
        }
        else {
            G_plus_upper[y + ny * x + ny * nbin * i] =
                G_plus_func(w_0_up, g0_up, epsi, epsilon2, mu_star, E_up);
            G_plus_lower[y + ny * x + ny * nbin * i] =
                G_plus_func(w_0_low, g0_low, epsi, epsilon2, mu_star, E_low);
            G_minus_upper[y + ny * x + ny * nbin * i] =
                G_minus_func(w_0_up, g0_up, epsi, epsilon2, mu_star, E_up);
            G_minus_lower[y + ny * x + ny * nbin * i] =
                G_minus_func(w_0_low, g0_low, epsi, epsilon2, mu_star, E_low);
        }
    }
}
