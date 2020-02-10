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
    const int&    real_star,        // pil
    const double& fake_opac,        // io
    const double& T_surf,           // csp, cse, pil
    const double& surf_albedo,      // cse
    const bool&   correct_surface_emissions,
    const bool&   interp_and_calc_flux_step) {
  int nbin = Alf_ptr->opacities.nbin;
  int nlayer = Alf_ptr->nlayer;
  int ninterface = Alf_ptr->ninterface;
  
  dim3 calc_surf_grid(int((nbin + 15) / 16), 1, 1);
    dim3 calc_surf_blocks(16, 1, 1);
    // csp

    // out: csp, cse 
    double* dev_planckband_lay = *(Alf_ptr->planckband_lay);
    // in: pil, pii
    double* dev_planckband_grid = *(Alf_ptr->plancktable.planck_grid);
    // out: pii
    double* dev_planckband_int = *(Alf_ptr->planckband_int);
    int plancktable_dim = Alf_ptr->plancktable.dim;
    int plancktable_step = Alf_ptr->plancktable.step;

    bool iso = Alf_ptr->iso;

    
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


bool wrap_prepare_compute_flux(
			  long dev_starflux, // pil
			  long dev_T_lay, // it, pil, io, mmm, kil
			  long dev_T_int, // it, pii, ioi, mmmi, kii
			  long dev_p_lay, // io, mmm, kil
			  long dev_p_int, // ioi, mmmi, kii
			  long dev_opac_wg_lay, // io
			  long dev_opac_wg_int, // ioi
			  long dev_meanmolmass_lay, // mmm
			  long dev_meanmolmass_int, // mmmi
			  const int & real_star, // pil
			  const double & fake_opac, // io
			  const double & T_surf, // csp, cse, pil
			  const double & surf_albedo, // cse
			  const bool & correct_surface_emissions,
			  const bool & interp_and_calc_flux_step
			  
			       )
{

  bool ret = prepare_compute_flux(
				  (double *)dev_starflux, // pil
				  (double *)dev_T_lay, // it, pil, io, mmm, kil
				  (double *)dev_T_int, // it, pii, ioi, mmmi, kii
				  (double *)dev_p_lay, // io, mmm, kil
				  (double *)dev_p_int, // ioi, mmmi, kii
				  (double *)dev_opac_wg_lay, // io
				  (double *)dev_opac_wg_int, // ioi
				  (double *)dev_meanmolmass_lay, // mmm
				  (double *)dev_meanmolmass_int, // mmmi
				  real_star, // pil
				  fake_opac, // io
				  T_surf, // csp, cse, pil
				  surf_albedo, // cse
				  correct_surface_emissions,
				  interp_and_calc_flux_step
				  );
    return ret;
}

void wrap_integrate_flux(long deltalambda,  // double*
                         long F_down_tot,   // double *
                         long F_up_tot,     // double *
                         long F_net,        // double *
                         long F_down_wg,    // double *
                         long F_up_wg,      // double *
                         long F_dir_wg,     // double *
                         long F_down_band,  // double *
                         long F_up_band,    // double *
                         long F_dir_band,   // double *
                         long gauss_weight) // double *
{
  integrate_flux((double*) deltalambda,
		 (double*) F_down_tot,
		 (double*) F_up_tot,
		 (double*) F_net,
		 (double*) F_down_wg,
		 (double*) F_up_wg,
		 (double*) F_dir_wg,
		 (double*) F_down_band,
		 (double*) F_up_band,
		 (double*) F_dir_band,
		 (double*) gauss_weight);

}

void integrate_flux(double* deltalambda,
		    double* F_down_tot,
		    double* F_up_tot,
		    double* F_net,
		    double* F_down_wg,
		    double* F_up_wg,
		    double* F_dir_wg,
		    double* F_down_band,
		    double* F_up_band,
		    double* F_dir_band,
		    double* gauss_weight)
{
  
    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(32, 4, 8);
    
    int nbin = Alf_ptr->opacities.nbin;
    int ny = Alf_ptr->opacities.ny;
    
    int ninterface = Alf_ptr->ninterface;
    
  
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
                                                          ninterface,
                                                          ny);

     cudaDeviceSynchronize();
}

__host__ bool calculate_transmission_iso(double* trans_wg,        // out
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
                                         int     clouds,
                                         int     scat_corr) {
  int nbin = Alf_ptr->opacities.nbin;
  int nlayer = Alf_ptr->nlayer;

    int ny = Alf_ptr->opacities.ny;
    

    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
    dim3 block(16, 4, 4);
    trans_iso<<<grid, block>>>(trans_wg,
                               *(Alf_ptr->delta_tau_wg),
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
                               Alf_ptr->nlayer,
                               clouds,
                               scat_corr);

    cudaDeviceSynchronize();
    return true;
}

bool wrap_calculate_transmission_iso(long   trans_wg,
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
                                     int    clouds,
                                     int    scat_corr) {
    return calculate_transmission_iso((double*)trans_wg,
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
                                      clouds,
                                      scat_corr);
}

__host__ bool calculate_transmission_noniso(double* trans_wg_upper,
                                            double* trans_wg_lower,
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
                                            int     clouds,
                                            int     scat_corr) {
  int nbin = Alf_ptr->opacities.nbin;
  int nlayer = Alf_ptr->nlayer;
  int ny = Alf_ptr->opacities.ny;
    
    

    
    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
    dim3 block(16, 4, 4);

    trans_noniso<<<grid, block>>>(trans_wg_upper,
                                  trans_wg_lower,
                                  *(Alf_ptr->delta_tau_wg_upper),
				  *(Alf_ptr->delta_tau_wg_lower),
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
                                  Alf_ptr->nlayer,
                                  clouds,
                                  scat_corr);
    cudaDeviceSynchronize();
    return true;
}

bool wrap_calculate_transmission_noniso(long   trans_wg_upper,
                                        long   trans_wg_lower,
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
                                        int    clouds,
                                        int    scat_corr) {
    return calculate_transmission_noniso((double*)trans_wg_upper,
                                         (double*)trans_wg_lower,
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
                                         clouds,
                                         scat_corr);
}


bool direct_beam_flux(double* F_dir_wg,
                      double* Fc_dir_wg,
                      double* z_lay,
                      double  mu_star,
                      double  R_planet,
                      double  R_star,
                      double  a,
                      int     dir_beam,
                      int     geom_zenith_corr) {

  double* planckband_lay = *(Alf_ptr->planckband_lay);
  int nbin = Alf_ptr->opacities.nbin;
  
  int ny = Alf_ptr->opacities.ny;
  
  int ninterface = Alf_ptr->ninterface;
  
    if (Alf_ptr->iso) {
        dim3 block(4, 32, 4);
        dim3 grid(int((ninterface + 3) / 4), int((nbin + 31) / 32), int((ny + 3) / 4));
        fdir_iso<<<grid, block>>>(F_dir_wg,
                                  planckband_lay,
                                  *(Alf_ptr->delta_tau_wg),
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
                                     *(Alf_ptr->delta_tau_wg_upper),
				     *(Alf_ptr->delta_tau_wg_lower),
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
                           long   z_lay,
                           double mu_star,
                           double R_planet,
                           double R_star,
                           double a,
                           int    dir_beam,
                           int    geom_zenith_corr) {    
    return direct_beam_flux((double*)F_dir_wg,
                            (double*)Fc_dir_wg,
                            (double*)z_lay,
                            mu_star,
                            R_planet,
                            R_star,
                            a,
                            dir_beam,
                            geom_zenith_corr);
}


bool populate_spectral_flux_iso(double* F_down_wg,    // out
                                double* F_up_wg,      // out
                                double* F_dir_wg,     // in
                                double* g_0_tot_lay,   // in
                                double  g_0,
                                int     singlewalk,
                                double  Rstar,
                                double  a,
                                double  f_factor,
                                double  mu_star,
                                double  epsi,
                                double  w_0_limit,
                                int     dir_beam,
                                int     clouds,
                                double  albedo) {
    double* planckband_lay = *(Alf_ptr->planckband_lay);
    int nbin = Alf_ptr->opacities.nbin;

    int ny = Alf_ptr->opacities.ny;
    
    int ninterface = Alf_ptr->ninterface;
    
    dim3 block(16, 16, 1);
    dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);
    fband_iso_notabu<<<grid, block>>>(F_down_wg,
                                      F_up_wg,
                                      F_dir_wg,
                                      planckband_lay,
                                      *(Alf_ptr->w_0),
                                      *(Alf_ptr->delta_tau_wg),
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
                                      ninterface,
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
                                     long   g_0_tot_lay,
                                     double g_0,
                                     int    singlewalk,
                                     double Rstar,
                                     double a,
                                     double f_factor,
                                     double mu_star,
                                     double epsi,
                                     double w_0_limit,
                                     int    dir_beam,
                                     int    clouds,
                                     double albedo) {
    return populate_spectral_flux_iso((double*)F_down_wg,
                                      (double*)F_up_wg,
                                      (double*)F_dir_wg,
                                      (double*)g_0_tot_lay,
                                      g_0,
                                      singlewalk,
                                      Rstar,
                                      a,
                                      f_factor,
                                      mu_star,
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
                                            double* g_0_tot_lay,
                                            double* g_0_tot_int,
                                            double  g_0,
                                            int     singlewalk,
                                            double  Rstar,
                                            double  a,
                                            double  f_factor,
                                            double  mu_star,
                                            double  epsi,
                                            double  w_0_limit,
                                            double  delta_tau_limit,
                                            int     dir_beam,
                                            int     clouds,
                                            double  albedo,
                                            double* trans_wg_upper,
                                            double* trans_wg_lower) {
  int nbin = Alf_ptr->opacities.nbin;
  int ny = Alf_ptr->opacities.ny;
    
  int ninterface = Alf_ptr->ninterface;
    

  dim3 block(16, 16, 1);
  
  dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);
  
    double* planckband_lay = *(Alf_ptr->planckband_lay);
    double* planckband_int = *(Alf_ptr->planckband_int);
 
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
                                         *(Alf_ptr->delta_tau_wg_upper),
					 *(Alf_ptr->delta_tau_wg_lower),
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
                                         ninterface,
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
                                        long   g_0_tot_lay,
                                        long   g_0_tot_int,
                                        double g_0,
                                        int    singlewalk,
                                        double Rstar,
                                        double a,
                                        double f_factor,
                                        double mu_star,
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
                                         (double*)g_0_tot_lay,
                                         (double*)g_0_tot_int,
                                         g_0,
                                         singlewalk,
                                         Rstar,
                                         a,
                                         f_factor,
                                         mu_star,
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

void init_parameters(const int& nlayer_,
		     const bool& iso_,
		     const double & Tstar_) {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return;
    }

    Alf_ptr->set_parameters(nlayer_, iso_, Tstar_);
}

void allocate() {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return;
    }

    Alf_ptr->allocate_internal_variables();
}

// TODO: this is ugly and should not exist!
std::tuple<long, 
	   long, long, long,
	   long, long, long,
	   long, long, long,
	   long, long, long,
	   long, long, long,
	   int, int> get_device_pointers_for_helios_write() {
    if (Alf_ptr == nullptr) {
        printf("ERROR: Alfrodull Engine not initialised");
        return std::make_tuple(0, 
			       0, 0, 0,
			       0, 0, 0,
			       0, 0, 0,
			       0, 0, 0,
			       0, 0, 0,
			       0, 0);
    }
    

    return Alf_ptr->get_device_pointers_for_helios_write( );
}

std::tuple<long,
	   long,
	   int,
	   int>
get_opac_data_for_helios() {
  if (Alf_ptr == nullptr) {
    printf("ERROR: Alfrodull Engine not initialised");
    return std::make_tuple(0, 0, 0, 0);
  }
  
  return Alf_ptr->get_opac_data_for_helios();
}


void prepare_planck_table()
{
  printf("Preparing planck table\n");
  if (Alf_ptr != nullptr)
    Alf_ptr->prepare_planck_table();
  else
    printf("ERROR: prepare_planck_table: no Alf_ptr\n");
}

void correct_incident_energy(long starflux_array_ptr,
			     bool real_star,
			     bool energy_budget_correction)
{
  printf("Correcting incident energy\n");

  if (Alf_ptr != nullptr)
    Alf_ptr->correct_incident_energy((double*)starflux_array_ptr, real_star, energy_budget_correction);
  else
    printf("ERROR: correct_incident_energy : no Alf_ptr\n");
}


void set_z_calc_function(std::function<void()> & func)
{
  if (Alf_ptr != nullptr)
    Alf_ptr->set_z_calc_func(func);
  
}



// var already present:
// bool iso 
void compute_radiative_transfer(
				// prepare_compute_flux
				
				// TODO: planck value tabulated and then interpolated
				double* dev_starflux,              // in: pil
				// state variables
				// TODO: check which ones can be internal only
				double*       dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
				double*       dev_T_int,           // in: it, pii, ioi, mmmi, kii  
				double*       dev_p_lay,           // in: io, mmm, kil
				double*       dev_p_int,           // in: ioi, mmmi, kii
				const int&    real_star,        // pil
				const double& fake_opac,        // io
				const double& T_surf,           // csp, cse, pil
				const double& surf_albedo,      // cse
				const bool&   correct_surface_emissions,
				const bool&   interp_and_calc_flux_step,
				// calculate_transmission_iso
				//double* trans_wg,        // out
				//double* opac_wg_lay,     // in
				//double* cloud_opac_lay,  // in
				//double* meanmolmass_lay, // in
				//double* cloud_scat_cross_lay, // in
				//double* g_0_tot_lay,       // in
				//double  g_0,
				//double  epsi,
				//double  mu_star,
				//int     scat,
				//int     ny,
				//int     clouds,
				//int     scat_corr
				
				
				// calculate_transmission_non_iso
				// double* trans_wg,        // out
				// double* trans_wg_upper,
				// double* trans_wg_lower,
				double* cloud_opac_lay,
				double* cloud_opac_int,
				double* cloud_scat_cross_lay,
				double* cloud_scat_cross_int,
				double* g_0_tot_lay,
				double* g_0_tot_int,
				double  g_0,
				double  epsi,
				double  mu_star,
				int     scat,
				int     clouds,
				int     scat_corr,
				// direct_beam_flux
				//double* F_dir_wg,
				//double* Fc_dir_wg,
				double* z_lay,
				//double  mu_star,
				double  R_planet,
				double  R_star,
				double  a,
				int     dir_beam,
				int     geom_zenith_corr,
				//int     ny
				
				// spectral flux loop
				bool single_walk,
				// int scat_val, -> same as scat
				
				
				// populate_spectral_flux_iso
				//double* F_down_wg,    // out
                                //double* F_up_wg,      // out
                                //double* F_dir_wg,     // in
                                //double* g_0_tot_lay,   // in
                                //double  g_0,
                                // int     singlewalk, -> single_walk 
				//                                double  Rstar, -> R_star
                                //double  a,
                                // int     numinterfaces, -> ninterface
                                double  f_factor,
                                //double  mu_star,
                                //int     ny,
                                //double  epsi,
                                double  w_0_limit,
                                double  albedo,
				// populate_spectral_flux_noniso
				double* F_down_wg,
				double* F_up_wg,
				double* Fc_down_wg,
				double* Fc_up_wg,
				double* F_dir_wg,
				double* Fc_dir_wg,
				//double* g_0_tot_lay,
				//double* g_0_tot_int,
				//double  g_0,
				//int     singlewalk,
				//double  Rstar,
				//double  a,
				//int     numinterfaces,
				//double  f_factor,
				//double  mu_star,
				//int     ny,
				//double  epsi,
				//double  w_0_limit,
				double  delta_tau_limit,
				//int     dir_beam,
				//int     clouds,
				//double  albedo, -> surf_albedo
				// double* trans_wg_upper,
				// double* trans_wg_lower,
				
				// integrate_flux
				double* F_down_tot,
				double* F_up_tot,
				double* F_net,
				//double* F_down_wg,
				//double* F_up_wg,
				//double* F_dir_wg,
				double* F_down_band,
				double* F_up_band,
				double* F_dir_band,
				double* gauss_weight
				//int num_interfaces, -> ninterface
				//int ny
				)
{

  double* delta_colmass = *Alf_ptr->delta_col_mass;
  double* delta_col_upper = *Alf_ptr->delta_col_upper;
  double* delta_col_lower = *Alf_ptr->delta_col_lower;

  double*       opac_wg_lay = *Alf_ptr->opac_wg_lay;     // out: io
  double*       opac_wg_int = *Alf_ptr->opac_wg_int;     // out: ioi
  double*       meanmolmass_lay = *Alf_ptr->meanmolmass_lay; // out: mmm
  double*       meanmolmass_int = *Alf_ptr->meanmolmass_int; // out: mmmi
  double* trans_wg = *Alf_ptr->trans_wg;        // out
  double* trans_wg_upper = *Alf_ptr->trans_wg_upper;
  double* trans_wg_lower = *Alf_ptr->trans_wg_lower;

  double* deltalambda = *Alf_ptr->opacities.dev_opac_deltawave;

  
  prepare_compute_flux(    dev_starflux,        
			   dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
			   dev_T_int,           // in: it, pii, ioi, mmmi, kii  
			   dev_p_lay,           // in: io, mmm, kil
			   dev_p_int,           // in: ioi, mmmi, kii
			   opac_wg_lay,     // out: io
			   opac_wg_int,     // out: ioi
			   meanmolmass_lay, // out: mmm
			   meanmolmass_int, // out: mmmi
			   real_star,        // pil
			   fake_opac,        // io
			   T_surf,           // csp, cse, pil
			   surf_albedo,      // cse
			   correct_surface_emissions,
			   interp_and_calc_flux_step);
    
    
    if (interp_and_calc_flux_step)
      {
	if (Alf_ptr->iso)
	  {
	    calculate_transmission_iso( trans_wg,        // out
					delta_colmass,   // in
					opac_wg_lay,     // in
					cloud_opac_lay,  // in
					meanmolmass_lay, // in
					cloud_scat_cross_lay, // in
					g_0_tot_lay,       // in
					g_0,
					epsi,
					mu_star,
					scat,
					clouds,
					scat_corr);
	  }
	else
	  {
	    calculate_transmission_noniso( trans_wg_upper,
					   trans_wg_lower,
					   delta_col_upper,
					   delta_col_lower,
					   opac_wg_lay,
					   opac_wg_int,
					   cloud_opac_lay,
					   cloud_opac_int,
					   meanmolmass_lay,
					   meanmolmass_int,
					   cloud_scat_cross_lay,
					   cloud_scat_cross_int,
					   g_0_tot_lay,
					   g_0_tot_int,
					   g_0,
					   epsi,
					   mu_star,
					   scat,
					   clouds,
					   scat_corr);
	  }
	
	
	Alf_ptr->call_z_callback();
	
	direct_beam_flux( F_dir_wg,
			  Fc_dir_wg,
			  z_lay,
			  mu_star,
			  R_planet,
			  R_star,
			  a,
			  dir_beam,
                          geom_zenith_corr);
	  }
  
  int nscat_step = 0;
  if (single_walk)
    nscat_step = 200;
  else
    nscat_step = 3;
  
  for (int scat_iter = 0; scat_iter < nscat_step*scat + 1; scat_iter++)
    {
      if (Alf_ptr->iso)
	{
	  populate_spectral_flux_iso( F_down_wg,    // out
				      F_up_wg,      // out
				      F_dir_wg,     // in
				      g_0_tot_lay,   // in
				      g_0,
				      single_walk,
				      R_star,
				      a,
				      f_factor,
				      mu_star,
				      epsi,
				      w_0_limit,
				      dir_beam,
				      clouds,
				      albedo);
	}
      else
	{
	  populate_spectral_flux_noniso( F_down_wg,
					 F_up_wg,
					 Fc_down_wg,
					 Fc_up_wg,
					 F_dir_wg,
					 Fc_dir_wg,
					 g_0_tot_lay,
					 g_0_tot_int,
					 g_0,
					 single_walk,
					 R_star,
					 a,
					 f_factor,
					 mu_star,
					 epsi,
					 w_0_limit,
					 delta_tau_limit,
					 dir_beam,
					 clouds,
					 albedo,
					 trans_wg_upper,
					 trans_wg_lower);
	}
      
    }
  integrate_flux( deltalambda,
		  F_down_tot,
		  F_up_tot,
		  F_net,
		  F_down_wg,
		  F_up_wg,
		  F_dir_wg,
		  F_down_band,
		  F_up_band,
		  F_dir_band,
		  gauss_weight);


  cudaError_t err = cudaGetLastError();
 
  if (err != cudaSuccess) {
    printf("compute_radiative_transfer: cuda error: %s\n", cudaGetErrorString(err));
  }
}

void wrap_compute_radiative_transfer(
				     // prepare_compute_flux
				     long dev_starflux,              // in: pil
				     // state variables
				     long       dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
				     long       dev_T_int,           // in: it, pii, ioi, mmmi, kii  
				     long       dev_p_lay,           // in: io, mmm, kil
				     long       dev_p_int,           // in: ioi, mmmi, kii
				     const int&    real_star,        // pil
				     const double& fake_opac,        // io
				     const double& T_surf,           // csp, cse, pil
				     const double& surf_albedo,      // cse
				     const bool&   correct_surface_emissions,
				     const bool&   interp_and_calc_flux_step,				     
				     // calculate_transmission_non_iso
				     long cloud_opac_lay,
				     long cloud_opac_int,
				     long cloud_scat_cross_lay,
				     long cloud_scat_cross_int,
				     long g_0_tot_lay,
				     long g_0_tot_int,
				     double  g_0,
				     double  epsi,
				     double  mu_star,
				     int     scat,
				     int     clouds,
				     int     scat_corr,
				     // direct_beam_flux
				     long z_lay,
				     double  R_planet,
				     double  R_star,
				     double  a,
				     int     dir_beam,
				     int     geom_zenith_corr,
				     // spectral flux loop
				     bool single_walk,
				     // populate_spectral_flux_iso
				     double  f_factor,
				     double  w_0_limit,
				     double  albedo,
				     // populate_spectral_flux_noniso
				     long F_down_wg,
				     long F_up_wg,
				     long Fc_down_wg,
				     long Fc_up_wg,
				     long F_dir_wg,
				     long Fc_dir_wg,
				     double  delta_tau_limit,
				     // integrate_flux
				     long F_down_tot,
				     long F_up_tot,
				     long F_net,
				     long F_down_band,
				     long F_up_band,
				     long F_dir_band,
				     long gauss_weight
				     )
{
  compute_radiative_transfer(
			     (double*) dev_starflux,              // in: pil
			     (double*)       dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
			     (double*)       dev_T_int,           // in: it, pii, ioi, mmmi, kii  
			     (double*)       dev_p_lay,           // in: io, mmm, kil
			     (double*)       dev_p_int,           // in: ioi, mmmi, kii
			     real_star,        // pil
			     fake_opac,        // io
			     T_surf,           // csp, cse, pil
			     surf_albedo,      // cse
			     correct_surface_emissions,
			     interp_and_calc_flux_step,
			     // calculate_transmission_non_iso
			     (double*) cloud_opac_lay,
			     (double*) cloud_opac_int,
			     (double*) cloud_scat_cross_lay,
			     (double*) cloud_scat_cross_int,
			     (double*) g_0_tot_lay,
			     (double*) g_0_tot_int,
			     g_0,
			     epsi,
			     mu_star,
			     scat,
			     clouds,
			     scat_corr,
			     // direct_beam_flux
			     (double*) z_lay,
			     R_planet,
			     R_star,
			     a,
			     dir_beam,
			     geom_zenith_corr,
			     // spectral flux loop
			     single_walk,
			     // populate_spectral_flux_iso
			     f_factor,
			     w_0_limit,
			     albedo,
			     // populate_spectral_flux_noniso
			     (double*) F_down_wg,
			     (double*) F_up_wg,
			     (double*) Fc_down_wg,
			     (double*) Fc_up_wg,
			     (double*) F_dir_wg,
			     (double*) Fc_dir_wg,
			     delta_tau_limit,
			     // integrate_flux
			     (double*) F_down_tot,
			     (double*) F_up_tot,
			     (double*) F_net,
			     (double*) F_down_band,
			     (double*) F_up_band,
			     (double*) F_dir_band,
			     (double*) gauss_weight
			     );
}
