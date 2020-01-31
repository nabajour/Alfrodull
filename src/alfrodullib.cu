#include "alfrodullib.h"

#include "integrate_flux.h"

#include "interpolate_values.h"

#include "surface_planck.h"

#include "correct_surface_emission.h"

#include "calculate_physics.h"

#include "alfrodull_engine.h"

#include <cstdio>
#include <memory>

std::unique_ptr<alfrodull_engine> Alf_ptr = nullptr;

__host__ bool prepare_compute_flux(
    // TODO: planck value tabulated and then interpolated
    double* dev_planckband_lay,        // out: csp, cse 
    double* dev_planckband_grid,       // in: pil, pii
    double* dev_planckband_int,        // out: pii
    double* dev_starflux,              // in: pil
    // state variables
    // TODO: check which ones can be internal only
    double*       dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    double*       dev_T_int,           // in: it, pii, ioi, mmmi, kii  
    double*       dev_p_lay,           // in: io, mmm, kil
    double*       dev_p_int,           // in: ioi, mmmi, kii
    double*       dev_opac_wg_lay,     // out: io
    double*       dev_opac_wg_int,     // out: ioi
    double*       dev_meanmolmass_lay, // out: mmm
    double*       dev_meanmolmass_int, // out: mmmi
    const int&    ninterface,          // it, pii, mmmi, kii
    const int&    nbin,                // csp, cse, pil, pii, io
    const int&    nlayer,              // csp, cse, pil, io, mmm, kil
    const int&    real_star,        // pil
    const double& fake_opac,        // io
    const double& T_surf,           // csp, cse, pil
    const double& surf_albedo,      // cse
    const int&    plancktable_dim,  // pil, pii
    const int&    plancktable_step, // pil, pii
    const bool&   iso,                // pii
    const bool&   correct_surface_emissions,
    const bool&   interp_and_calc_flux_step) {
    dim3 calc_surf_grid(int((nbin + 15) / 16), 1, 1);
    dim3 calc_surf_blocks(16, 1, 1);
    // csp
    calc_surface_planck<<<calc_surf_grid, calc_surf_blocks>>>(
							      dev_planckband_lay,                       // out
							      *(Alf_ptr->opacities.dev_opac_interwave), // in
							      *(Alf_ptr->opacities.dev_opac_deltawave), // in
							      nbin,
							      //Alf_ptr->opacities.nbin,
							      nlayer,
							      T_surf);
    
    cudaDeviceSynchronize();

    if (correct_surface_emissions) {
        dim3 corr_surf_emiss_grid(int((nbin + 15) / 16), 1, 1);
        dim3 corr_surf_emiss_block(16, 1, 1);
        // cse
        correct_surface_emission<<<corr_surf_emiss_grid, corr_surf_emiss_block>>>(
	    *(Alf_ptr->opacities.dev_opac_deltawave),     // in
            dev_planckband_lay,                           // in/out
            surf_albedo,
            T_surf,
	    nbin,
	    //Alf_ptr->opacities.nbin,
            nlayer);

        cudaDeviceSynchronize();
    }

    // it
    dim3 it_grid(int((ninterface + 15) / 16), 1, 1);
    dim3 it_block(16, 1, 1);

    interpolate_temperature<<<it_grid, it_block>>>(dev_T_lay,   // out
						   dev_T_int,   // in
						   ninterface);
    cudaDeviceSynchronize();

    // pil
    dim3 pil_grid(int((nbin + 15) / 16), int(((nlayer + 2) + 15)) / 16, 1);
    dim3 pil_block(16, 16, 1);
    planck_interpol_layer<<<pil_grid, pil_block>>>(dev_T_lay,           // in
                                                   dev_planckband_lay,  // out
                                                   dev_planckband_grid, // in
                                                   dev_starflux,        // in
                                                   real_star,
                                                   nlayer,
                                                   nbin,
                                                   T_surf,
                                                   plancktable_dim,
                                                   plancktable_step);
    cudaDeviceSynchronize();

    if (!iso) {
        // pii
        dim3 pii_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), 1);
        dim3 pii_block(16, 16, 1);
        planck_interpol_interface<<<pii_grid, pii_block>>>(dev_T_int,            // in
                                                           dev_planckband_int,   // out
                                                           dev_planckband_grid,  // in
                                                           ninterface,
                                                           nbin,
                                                           plancktable_dim,
                                                           plancktable_step);
        cudaDeviceSynchronize();
    }

    if (interp_and_calc_flux_step) {
        // io
        dim3 io_grid(int((nbin + 15) / 16), int((nlayer + 15) / 16), 1);
        dim3 io_block(16, 16, 1);
        // TODO: should move fake_opac (opacity limit somewhere into opacity_table/interpolation component?)
	// out -> opacities (dev_opac_wg_lay)
	// out -> scetter cross section (scatter_cross_section_...)
        interpolate_opacities<<<io_grid, io_block>>>(dev_T_lay,                                     // in
                                                     *(Alf_ptr->opacities.dev_temperatures),        // in
                                                     dev_p_lay,                                     // in
                                                     *(Alf_ptr->opacities.dev_pressures),           // in
                                                     *(Alf_ptr->opacities.dev_kpoints),             // in
                                                     dev_opac_wg_lay,                               // out
                                                     *(Alf_ptr->opacities.dev_scat_cross_sections), // in
                                                     *(Alf_ptr->scatter_cross_section_lay),         // out
                                                     Alf_ptr->opacities.n_pressures,
                                                     Alf_ptr->opacities.n_temperatures,
                                                     Alf_ptr->opacities.ny,
                                                     nbin,
                                                     fake_opac,
                                                     nlayer);


        cudaDeviceSynchronize();

        if (!iso) {
            // ioi
            dim3 ioi_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), 1);
            dim3 ioi_block(16, 16, 1);

            interpolate_opacities<<<ioi_grid, ioi_block>>>(
							   dev_T_int,                                    // in
							   *(Alf_ptr->opacities.dev_temperatures),       // in
							   dev_p_int,                                    // in
							   *(Alf_ptr->opacities.dev_pressures),          // in
							   *(Alf_ptr->opacities.dev_kpoints),            // in
							   dev_opac_wg_int,                              // out
							   *(Alf_ptr->opacities.dev_scat_cross_sections),// in
							   *(Alf_ptr->scatter_cross_section_inter),      // out
							   Alf_ptr->opacities.n_pressures,
							   Alf_ptr->opacities.n_temperatures,
							   Alf_ptr->opacities.ny,
							   nbin,
							   fake_opac,
							   ninterface);
	    
            cudaDeviceSynchronize();
        }

        // mmm
        dim3 mmm_block(16, 1, 1);
        dim3 mmm_grid(int((nlayer + 15) / 16), 1, 1);

        meanmolmass_interpol<<<mmm_grid, mmm_block>>>(dev_T_lay,                              // in
                                                      *(Alf_ptr->opacities.dev_temperatures), // in 
                                                      dev_meanmolmass_lay,                    // out
                                                      *(Alf_ptr->opacities.dev_meanmolmass),  // in
                                                      dev_p_lay,                              // in
                                                      *(Alf_ptr->opacities.dev_pressures),    // in
                                                      Alf_ptr->opacities.n_pressures,
                                                      Alf_ptr->opacities.n_temperatures,
                                                      nlayer);


        cudaDeviceSynchronize();

        if (!iso) {
            // mmmi
            dim3 mmmi_block(16, 1, 1);
            dim3 mmmi_grid(int((ninterface + 15) / 16), 1, 1);

            meanmolmass_interpol<<<mmmi_grid, mmmi_block>>>(dev_T_int,                              // in
                                                            *(Alf_ptr->opacities.dev_temperatures), // in
                                                            dev_meanmolmass_int,                    // out
                                                            *(Alf_ptr->opacities.dev_meanmolmass),  // in
                                                            dev_p_int,                              // in
                                                            *(Alf_ptr->opacities.dev_pressures),    // in
                                                            Alf_ptr->opacities.n_pressures,
                                                            Alf_ptr->opacities.n_temperatures,
                                                            ninterface);


            cudaDeviceSynchronize();
        }

    }

    // TODO: add state check and return value

    return true;
}

void wrap_integrate_flux(long deltalambda_,  // double*
                         long F_down_tot_,   // double *
                         long F_up_tot_,     // double *
                         long F_net_,        // double *
                         long F_down_wg_,    // double *
                         long F_up_wg_,      // double *
                         long F_dir_wg_,     // double *
                         long F_down_band_,  // double *
                         long F_up_band_,    // double *
                         long F_dir_band_,   // double *
                         long gauss_weight_, // double *
                         int  nbin,
                         int  numinterfaces,
                         int  ny,
                         int  block_x,
                         int  block_y,
                         int  block_z,
                         int  grid_x,
                         int  grid_y,
                         int  grid_z) {
    double* deltalambda  = (double*)deltalambda_;
    double* F_down_tot   = (double*)F_down_tot_;
    double* F_up_tot     = (double*)F_up_tot_;
    double* F_net        = (double*)F_net_;
    double* F_down_wg    = (double*)F_down_wg_;
    double* F_up_wg      = (double*)F_up_wg_;
    double* F_dir_wg     = (double*)F_dir_wg_;
    double* F_down_band  = (double*)F_down_band_;
    double* F_up_band    = (double*)F_up_band_;
    double* F_dir_band   = (double*)F_dir_band_;
    double* gauss_weight = (double*)gauss_weight_;

    dim3 threadsPerBlock(grid_x, grid_y, grid_z);
    dim3 numBlocks(block_x, block_y, block_z);

    printf("Running Alfrodull Wrapper for integrate flux\n");
    integrate_flux_double<<<threadsPerBlock, numBlocks>>>(deltalambda,
                                                          F_down_tot,
                                                          F_up_tot,
                                                          F_net,
                                                          F_down_wg,
                                                          F_up_wg,
                                                          F_dir_wg,
                                                          F_down_band,
                                                          F_up_band,
                                                          F_dir_band,
                                                          gauss_weight,
                                                          nbin,
                                                          numinterfaces,
                                                          ny);

     cudaDeviceSynchronize();
}

__host__ bool calculate_transmission_iso(double* trans_wg,        // out
                                         double* delta_tau_wg,    // out 
                                         double* delta_colmass,   // in
                                         double* opac_wg_lay,     // in
                                         double* cloud_opac_lay,  // in
                                         double* meanmolmass_lay, // in
                                         double* cloud_scat_cross_lay, // in
                                         double* g_0_tot_lay,       // in
                                         double  g_0,
                                         double  epsi,
                                         double  mu_star,
                                         int     scat,
                                         int     nbin,
                                         int     ny,
                                         int     nlayer,
                                         int     clouds,
                                         int     scat_corr) {
    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
    dim3 block(16, 4, 4);
    trans_iso<<<grid, block>>>(trans_wg,
                               delta_tau_wg,
                               *(Alf_ptr->M_term),
                               *(Alf_ptr->N_term),
				 *(Alf_ptr->P_term),
				 *(Alf_ptr->G_plus),
				 *(Alf_ptr->G_minus),
                               delta_colmass,
                               opac_wg_lay,
                               cloud_opac_lay,
                               meanmolmass_lay,
                               *(Alf_ptr->scatter_cross_section_lay),
                               cloud_scat_cross_lay,
				 *(Alf_ptr->w_0),
                               g_0_tot_lay,
                               g_0,
                               epsi,
                               mu_star,
                               scat,
                               nbin,
                               ny,
                               nlayer,
                               clouds,
                               scat_corr);

    cudaDeviceSynchronize();
    return true;
}

bool wrap_calculate_transmission_iso(long   trans_wg,
                                     long   delta_tau_wg,
                                     long   delta_colmass,
                                     long   opac_wg_lay,
                                     long   cloud_opac_lay,
                                     long   meanmolmass_lay,
                                     long   cloud_scat_cross_lay,
                                     long   g_0_tot_lay,
                                     double g_0,
                                     double epsi,
                                     double mu_star,
                                     int    scat,
                                     int    nbin,
                                     int    ny,
                                     int    nlayer,
                                     int    clouds,
                                     int    scat_corr) {
    return calculate_transmission_iso((double*)trans_wg,
                                      (double*)delta_tau_wg,
                                      (double*)delta_colmass,
                                      (double*)opac_wg_lay,
                                      (double*)cloud_opac_lay,
                                      (double*)meanmolmass_lay,
                                      (double*)cloud_scat_cross_lay,
                                      (double*)g_0_tot_lay,
                                      g_0,
                                      epsi,
                                      mu_star,
                                      scat,
                                      nbin,
                                      ny,
                                      nlayer,
                                      clouds,
                                      scat_corr);
}

__host__ bool calculate_transmission_noniso(double* trans_wg_upper,
                                            double* trans_wg_lower,
                                            double* delta_tau_wg_upper,
                                            double* delta_tau_wg_lower,
                                            double* delta_col_upper,
                                            double* delta_col_lower,
                                            double* opac_wg_lay,
                                            double* opac_wg_int,
                                            double* cloud_opac_lay,
                                            double* cloud_opac_int,
                                            double* meanmolmass_lay,
                                            double* meanmolmass_int,
                                            double* cloud_scat_cross_lay,
                                            double* cloud_scat_cross_int,
                                            double* g_0_tot_lay,
                                            double* g_0_tot_int,
                                            double  g_0,
                                            double  epsi,
                                            double  mu_star,
                                            int     scat,
                                            int     nbin,
                                            int     ny,
                                            int     nlayer,
                                            int     clouds,
                                            int     scat_corr) {
    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
    dim3 block(16, 4, 4);

    trans_noniso<<<grid, block>>>(trans_wg_upper,
                                  trans_wg_lower,
                                  delta_tau_wg_upper,
                                  delta_tau_wg_lower,
                                  *(Alf_ptr->M_upper),
                                  *(Alf_ptr->M_lower),
                                  *(Alf_ptr->N_upper),
                                  *(Alf_ptr->N_lower),
                                  *(Alf_ptr->P_upper),
                                  *(Alf_ptr->P_lower),
                                  *(Alf_ptr->G_plus_upper),
                                  *(Alf_ptr->G_plus_lower),
                                  *(Alf_ptr->G_minus_upper),
                                  *(Alf_ptr->G_minus_lower),
                                  delta_col_upper,
                                  delta_col_lower,
                                  opac_wg_lay,
                                  opac_wg_int,
                                  cloud_opac_lay,
                                  cloud_opac_int,
                                  meanmolmass_lay,
                                  meanmolmass_int,
                                  *(Alf_ptr->scatter_cross_section_lay),
                                  *(Alf_ptr->scatter_cross_section_inter),
                                  cloud_scat_cross_lay,
                                  cloud_scat_cross_int,
                                  *(Alf_ptr->w_0_upper),
				  *(Alf_ptr->w_0_lower),
                                  g_0_tot_lay,
                                  g_0_tot_int,
                                  g_0,
                                  epsi,
                                  mu_star,
                                  scat,
                                  nbin,
                                  ny,
                                  nlayer,
                                  clouds,
                                  scat_corr);
    cudaDeviceSynchronize();
    return true;
}

bool wrap_calculate_transmission_noniso(long   trans_wg_upper,
                                        long   trans_wg_lower,
                                        long   delta_tau_wg_upper,
                                        long   delta_tau_wg_lower,
                                        long   delta_col_upper,
                                        long   delta_col_lower,
                                        long   opac_wg_lay,
                                        long   opac_wg_int,
                                        long   cloud_opac_lay,
                                        long   cloud_opac_int,
                                        long   meanmolmass_lay,
                                        long   meanmolmass_int,
                                        long   cloud_scat_cross_lay,
                                        long   cloud_scat_cross_int,
                                        long   g_0_tot_lay,
                                        long   g_0_tot_int,
                                        double g_0,
                                        double epsi,
                                        double mu_star,
                                        int    scat,
                                        int    nbin,
                                        int    ny,
                                        int    nlayer,
                                        int    clouds,
                                        int    scat_corr) {
    return calculate_transmission_noniso((double*)trans_wg_upper,
                                         (double*)trans_wg_lower,
                                         (double*)delta_tau_wg_upper,
                                         (double*)delta_tau_wg_lower,
                                         (double*)delta_col_upper,
                                         (double*)delta_col_lower,
                                         (double*)opac_wg_lay,
                                         (double*)opac_wg_int,
                                         (double*)cloud_opac_lay,
                                         (double*)cloud_opac_int,
                                         (double*)meanmolmass_lay,
                                         (double*)meanmolmass_int,
                                         (double*)cloud_scat_cross_lay,
                                         (double*)cloud_scat_cross_int,
                                         (double*)g_0_tot_lay,
                                         (double*)g_0_tot_int,
                                         g_0,
                                         epsi,
                                         mu_star,
                                         scat,
                                         nbin,
                                         ny,
                                         nlayer,
                                         clouds,
                                         scat_corr);
}


bool direct_beam_flux(double* F_dir_wg,
                      double* Fc_dir_wg,
                      double* planckband_lay,
                      double* delta_tau_wg,
                      double* delta_tau_wg_upper,
                      double* delta_tau_wg_lower,
                      double* z_lay,
                      double  mu_star,
                      double  R_planet,
                      double  R_star,
                      double  a,
                      int     dir_beam,
                      int     geom_zenith_corr,
                      int     ninterface,
                      int     nbin,
                      int     ny,
                      bool    iso) {
    if (iso) {
        dim3 block(4, 32, 4);
        dim3 grid(int((ninterface + 3) / 4), int((nbin + 31) / 32), int((ny + 3) / 4));
        fdir_iso<<<grid, block>>>(F_dir_wg,
                                  planckband_lay,
                                  delta_tau_wg,
                                  z_lay,
                                  mu_star,
                                  R_planet,
                                  R_star,
                                  a,
                                  dir_beam,
                                  geom_zenith_corr,
                                  ninterface,
                                  nbin,
                                  ny);

        cudaDeviceSynchronize();
    }
    else {
        dim3 block(4, 32, 4);
        dim3 grid(int((ninterface + 3) / 4), int((nbin + 31) / 32), int((ny + 3) / 4));

        fdir_noniso<<<grid, block>>>(F_dir_wg,
                                     Fc_dir_wg,
                                     planckband_lay,
                                     delta_tau_wg_upper,
                                     delta_tau_wg_lower,
                                     z_lay,
                                     mu_star,
                                     R_planet,
                                     R_star,
                                     a,
                                     dir_beam,
                                     geom_zenith_corr,
                                     ninterface,
                                     nbin,
                                     ny);

        cudaDeviceSynchronize();
    }

    return true;
}

bool wrap_direct_beam_flux(long   F_dir_wg,
                           long   Fc_dir_wg,
                           long   planckband_lay,
                           long   delta_tau_wg,
                           long   delta_tau_wg_upper,
                           long   delta_tau_wg_lower,
                           long   z_lay,
                           double mu_star,
                           double R_planet,
                           double R_star,
                           double a,
                           int    dir_beam,
                           int    geom_zenith_corr,
                           int    ninterface,
                           int    nbin,
                           int    ny,
                           bool   iso) {
    return direct_beam_flux((double*)F_dir_wg,
                            (double*)Fc_dir_wg,
                            (double*)planckband_lay,
                            (double*)delta_tau_wg,
                            (double*)delta_tau_wg_upper,
                            (double*)delta_tau_wg_lower,
                            (double*)z_lay,
                            mu_star,
                            R_planet,
                            R_star,
                            a,
                            dir_beam,
                            geom_zenith_corr,
                            ninterface,
                            nbin,
                            ny,
                            iso);
}


bool populate_spectral_flux_iso(double* F_down_wg,    // out
                                double* F_up_wg,      // out
                                double* F_dir_wg,     // in
                                double* planckband_lay,  // in
                                double* delta_tau_wg,  // in
                                double* g_0_tot_lay,   // in
                                double  g_0,
                                int     singlewalk,
                                double  Rstar,
                                double  a,
                                int     numinterfaces,
                                int     nbin,
                                double  f_factor,
                                double  mu_star,
                                int     ny,
                                double  epsi,
                                double  w_0_limit,
                                int     dir_beam,
                                int     clouds,
                                double  albedo) {
    dim3 block(16, 16, 1);
    dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);
    fband_iso_notabu<<<grid, block>>>(F_down_wg,
                                      F_up_wg,
                                      F_dir_wg,
                                      planckband_lay,
                                      *(Alf_ptr->w_0),
                                      delta_tau_wg,
                                      *(Alf_ptr->M_term),
                                      *(Alf_ptr->N_term),
                                      *(Alf_ptr->P_term),
                                      *(Alf_ptr->G_plus),
                                      *(Alf_ptr->G_minus),
                                      g_0_tot_lay,
                                      g_0,
                                      singlewalk,
                                      Rstar,
                                      a,
                                      numinterfaces,
                                      nbin,
                                      f_factor,
                                      mu_star,
                                      ny,
                                      epsi,
                                      w_0_limit,
                                      dir_beam,
                                      clouds,
                                      albedo);

    return true;
}

bool wrap_populate_spectral_flux_iso(long   F_down_wg,
                                     long   F_up_wg,
                                     long   F_dir_wg,
                                     long   planckband_lay,
                                     long   delta_tau_wg,
                                     long   g_0_tot_lay,
                                     double g_0,
                                     int    singlewalk,
                                     double Rstar,
                                     double a,
                                     int    numinterfaces,
                                     int    nbin,
                                     double f_factor,
                                     double mu_star,
                                     int    ny,
                                     double epsi,
                                     double w_0_limit,
                                     int    dir_beam,
                                     int    clouds,
                                     double albedo) {
    return populate_spectral_flux_iso((double*)F_down_wg,
                                      (double*)F_up_wg,
                                      (double*)F_dir_wg,
                                      (double*)planckband_lay,
                                      (double*)delta_tau_wg,
                                      (double*)g_0_tot_lay,
                                      g_0,
                                      singlewalk,
                                      Rstar,
                                      a,
                                      numinterfaces,
                                      nbin,
                                      f_factor,
                                      mu_star,
                                      ny,
                                      epsi,
                                      w_0_limit,
                                      dir_beam,
                                      clouds,
                                      albedo);
}


// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__host__ bool populate_spectral_flux_noniso(double* F_down_wg,
                                            double* F_up_wg,
                                            double* Fc_down_wg,
                                            double* Fc_up_wg,
                                            double* F_dir_wg,
                                            double* Fc_dir_wg,
                                            double* planckband_lay,
                                            double* planckband_int,
                                            double* delta_tau_wg_upper,
                                            double* delta_tau_wg_lower,
                                            double* g_0_tot_lay,
                                            double* g_0_tot_int,
                                            double  g_0,
                                            int     singlewalk,
                                            double  Rstar,
                                            double  a,
                                            int     numinterfaces,
                                            int     nbin,
                                            double  f_factor,
                                            double  mu_star,
                                            int     ny,
                                            double  epsi,
                                            double  w_0_limit,
                                            double  delta_tau_limit,
                                            int     dir_beam,
                                            int     clouds,
                                            double  albedo,
                                            double* trans_wg_upper,
                                            double* trans_wg_lower) {
    dim3 block(16, 16, 1);
    dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);
    // calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
    fband_noniso_notabu<<<grid, block>>>(F_down_wg,
                                         F_up_wg,
                                         Fc_down_wg,
                                         Fc_up_wg,
                                         F_dir_wg,
                                         Fc_dir_wg,
                                         planckband_lay,
                                         planckband_int,
                                         *(Alf_ptr->w_0_upper),
                                         *(Alf_ptr->w_0_lower),
                                         delta_tau_wg_upper,
                                         delta_tau_wg_lower,
                                         *(Alf_ptr->M_upper),
                                         *(Alf_ptr->M_lower),
                                         *(Alf_ptr->N_upper),
                                         *(Alf_ptr->N_lower),
                                         *(Alf_ptr->P_upper),
                                         *(Alf_ptr->P_lower),
                                         *(Alf_ptr->G_plus_upper),
                                         *(Alf_ptr->G_plus_lower),
                                         *(Alf_ptr->G_minus_upper),
                                         *(Alf_ptr->G_minus_lower),
                                         g_0_tot_lay,
                                         g_0_tot_int,
                                         g_0,
                                         singlewalk,
                                         Rstar,
                                         a,
                                         numinterfaces,
                                         nbin,
                                         f_factor,
                                         mu_star,
                                         ny,
                                         epsi,
                                         w_0_limit,
                                         delta_tau_limit,
                                         dir_beam,
                                         clouds,
                                         albedo,
                                         trans_wg_upper,
                                         trans_wg_lower);

    return true;
}

bool wrap_populate_spectral_flux_noniso(long   F_down_wg,
                                        long   F_up_wg,
                                        long   Fc_down_wg,
                                        long   Fc_up_wg,
                                        long   F_dir_wg,
                                        long   Fc_dir_wg,
                                        long   planckband_lay,
                                        long   planckband_int,
                                        long   delta_tau_wg_upper,
                                        long   delta_tau_wg_lower,
                                        long   g_0_tot_lay,
                                        long   g_0_tot_int,
                                        double g_0,
                                        int    singlewalk,
                                        double Rstar,
                                        double a,
                                        int    numinterfaces,
                                        int    nbin,
                                        double f_factor,
                                        double mu_star,
                                        int    ny,
                                        double epsi,
                                        double w_0_limit,
                                        double delta_tau_limit,
                                        int    dir_beam,
                                        int    clouds,
                                        double albedo,
                                        long   trans_wg_upper,
                                        long   trans_wg_lower) {
    return populate_spectral_flux_noniso((double*)F_down_wg,
                                         (double*)F_up_wg,
                                         (double*)Fc_down_wg,
                                         (double*)Fc_up_wg,
                                         (double*)F_dir_wg,
                                         (double*)Fc_dir_wg,
                                         (double*)planckband_lay,
                                         (double*)planckband_int,
                                         (double*)delta_tau_wg_upper,
                                         (double*)delta_tau_wg_lower,
                                         (double*)g_0_tot_lay,
                                         (double*)g_0_tot_int,
                                         g_0,
                                         singlewalk,
                                         Rstar,
                                         a,
                                         numinterfaces,
                                         nbin,
                                         f_factor,
                                         mu_star,
                                         ny,
                                         epsi,
                                         w_0_limit,
                                         delta_tau_limit,
                                         dir_beam,
                                         clouds,
                                         albedo,
                                         (double*)trans_wg_upper,
                                         (double*)trans_wg_lower);
}


void init_alfrodull() {
    printf("Create Alfrodull Engine\n");

    Alf_ptr = std::make_unique<alfrodull_engine>();

    Alf_ptr->init();
}

void deinit_alfrodull() {
    cudaError_t err = cudaGetLastError();

    // Check device query
    if (err != cudaSuccess) {
        printf("deinit_alfrodull: cuda error: %s\n", cudaGetErrorString(err));
    }

    printf("Clean up Alfrodull Engine\n");
    Alf_ptr = nullptr;
}

void init_parameters(const int& nlayer_, const bool& iso_) {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return;
    }

    Alf_ptr->set_parameters(nlayer_, iso_);
}

void allocate() {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return;
    }

    Alf_ptr->allocate_internal_variables();
}

// TODO: this is ugly and should not exist!
std::tuple<long, long, long, long> get_device_pointers_for_helios_write() {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return std::make_tuple(0, 0, 0, 0);
    }
    double* dev_scat_cross_section_lay_ptr = 0;
    double* dev_scat_cross_section_int_ptr = 0;
    double* dev_interwave_ptr              = 0;
    double* dev_deltawave_ptr              = 0;

    Alf_ptr->get_device_pointers_for_helios_write(dev_scat_cross_section_lay_ptr,
                                                  dev_scat_cross_section_int_ptr,
                                                  dev_interwave_ptr,
                                                  dev_deltawave_ptr);

    long dev_scat_cross_section_lay = (long)dev_scat_cross_section_lay_ptr;
    long dev_scat_cross_section_int = (long)dev_scat_cross_section_int_ptr;
    long dev_interwave              = (long)dev_interwave_ptr;
    long dev_deltawave              = (long)dev_deltawave_ptr;

    return std::make_tuple(dev_scat_cross_section_int,
			   dev_scat_cross_section_lay,
			   dev_interwave,
			   dev_deltawave);
}
