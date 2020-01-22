#include "alfrodullib.h"

#include "integrate_flux.h"

#include "interpolate_values.h"

#include "surface_planck.h"

#include "correct_surface_emission.h"

#include <cstdio>


bool wrap_prepare_compute_flux(
			  long dev_planckband_lay,  // csp, cse
			  long dev_planckband_grid,  // pil, pii
			  long dev_planckband_int,  // pii
			  long dev_starflux, // pil
			  long dev_opac_interwave,  // csp
			  long dev_opac_deltawave,  // csp, cse
			  long dev_F_down_tot, // cse
			  long dev_T_lay, // it, pil, io, mmm, kil
			  long dev_T_int, // it, pii, ioi, mmmi, kii
			  long dev_ktemp, // io, mmm, mmmi
			  long dev_p_lay, // io, mmm, kil
			  long dev_p_int, // ioi, mmmi, kii
			  long dev_kpress, // io, mmm, mmmi
			  long dev_opac_k, // io
			  long dev_opac_wg_lay, // io
			  long dev_opac_wg_int, // ioi
			  long dev_opac_scat_cross, // io
			  long dev_scat_cross_lay, // io
			  long dev_scat_cross_int, // ioi
			  long dev_meanmolmass_lay, // mmm
			  long dev_meanmolmass_int, // mmmi
			  long dev_opac_meanmass, // mmm, mmmi
			  long dev_opac_kappa, // kil, kii
			  long dev_entr_temp, // kil, kii
			  long dev_entr_press, // kil, kii
			  long dev_kappa_lay, // kil
			  long dev_kappa_int, // kii
			  const int & ninterface, // it, pii, mmmi, kii
			  const int & nbin, // csp, cse, pil, pii, io
			  const int & nlayer, // csp, cse, pil, io, mmm, kil
			  const int & iter_value, // cse // TODO: check what this is for. Should maybe be external
			  const int & real_star, // pil
			  const int & npress, // io, mmm, mmmi
			  const int & ntemp, // io, mmm, mmmi
			  const int & ny, // io
			  const double & fake_opac, // io
			  const double & T_surf, // csp, cse, pil
			  const double & surf_albedo, // cse
			  const int & dim, // pil, pii
			  const int & step, // pil, pii
			  const bool & use_kappa_manual, // ki
			  const double & kappa_manual_value, // ki	     
			  const bool & iso, // pii
			  const bool & correct_surface_emissions,
			  const bool & interp_and_calc_flux_step
			  
			       )
{

  bool ret = prepare_compute_flux(
				  (double *)dev_planckband_lay,  // csp, cse
				  (double *)dev_planckband_grid,  // pil, pii
				  (double *)dev_planckband_int,  // pii
				  (double *)dev_starflux, // pil
				  (double *)dev_opac_interwave,  // csp
				  (double *)dev_opac_deltawave,  // csp, cse
				  (double *)dev_F_down_tot, // cse
				  (double *)dev_T_lay, // it, pil, io, mmm, kil
				  (double *)dev_T_int, // it, pii, ioi, mmmi, kii
				  (double *)dev_ktemp, // io, mmm, mmmi
				  (double *)dev_p_lay, // io, mmm, kil
				  (double *)dev_p_int, // ioi, mmmi, kii
				  (double *)dev_kpress, // io, mmm, mmmi
				  (double *)dev_opac_k, // io
				  (double *)dev_opac_wg_lay, // io
				  (double *)dev_opac_wg_int, // ioi
				  (double *)dev_opac_scat_cross, // io
				  (double *)dev_scat_cross_lay, // io
				  (double *)dev_scat_cross_int, // ioi
				  (double *)dev_meanmolmass_lay, // mmm
				  (double *)dev_meanmolmass_int, // mmmi
				  (double *)dev_opac_meanmass, // mmm, mmmi
				  (double *)dev_opac_kappa, // kil, kii
				  (double *)dev_entr_temp, // kil, kii
				  (double *)dev_entr_press, // kil, kii
				  (double *)dev_kappa_lay, // kil
				  (double *)dev_kappa_int, // kii
				  ninterface, // it, pii, mmmi, kii
				  nbin, // csp, cse, pil, pii, io
				  nlayer, // csp, cse, pil, io, mmm, kil
				  iter_value, // cse // TODO: check what this is for. Should maybe be external
				  real_star, // pil
				  npress, // io, mmm, mmmi
				  ntemp, // io, mmm, mmmi
				  ny, // io
				  fake_opac, // io
				  T_surf, // csp, cse, pil
				  surf_albedo, // cse
				  dim, // pil, pii
				  step, // pil, pii
				  use_kappa_manual, // ki
				  kappa_manual_value, // ki	     
				  iso, // pii
				  correct_surface_emissions,
				  interp_and_calc_flux_step
				  );
    return ret;
}

bool prepare_compute_flux(
		  double * dev_planckband_lay,  // csp, cse
		  double * dev_planckband_grid,  // pil, pii
		  double * dev_planckband_int,  // pii
		  double * dev_starflux, // pil
		  double * dev_opac_interwave,  // csp
		  double * dev_opac_deltawave,  // csp, cse
		  double * dev_F_down_tot, // cse
		  double * dev_T_lay, // it, pil, io, mmm, kil
		  double * dev_T_int, // it, pii, ioi, mmmi, kii
		  double * dev_ktemp, // io, mmm, mmmi
		  double * dev_p_lay, // io, mmm, kil
		  double * dev_p_int, // ioi, mmmi, kii
		  double * dev_kpress, // io, mmm, mmmi
		  double * dev_opac_k, // io
		  double * dev_opac_wg_lay, // io
		  double * dev_opac_wg_int, // ioi
		  double * dev_opac_scat_cross, // io
		  double * dev_scat_cross_lay, // io
		  double * dev_scat_cross_int, // ioi
		  double * dev_meanmolmass_lay, // mmm
		  double * dev_meanmolmass_int, // mmmi
		  double * dev_opac_meanmass, // mmm, mmmi
		  double * dev_opac_kappa, // kil, kii
		  double * dev_entr_temp, // kil, kii
		  double * dev_entr_press, // kil, kii
		  double * dev_kappa_lay, // kil
		  double * dev_kappa_int, // kii
		  const int & ninterface, // it, pii, mmmi, kii
		  const int & nbin, // csp, cse, pil, pii, io
		  const int & nlayer, // csp, cse, pil, io, mmm, kil
		  const int & iter_value, // cse // TODO: check what this is for. Should maybe be external
		  const int & real_star, // pil
		  const int & npress, // io, mmm, mmmi
		  const int & ntemp, // io, mmm, mmmi
		  const int & ny, // io
		  const int & entr_npress, // kii, kil
		  const int & entr_ntemp, // kii, kil		  
		  const double & fake_opac, // io
		  const double & T_surf, // csp, cse, pil
		  const double & surf_albedo, // cse
		  const int & plancktable_dim, // pil, pii
		  const int & plancktable_step, // pil, pii
		  const bool & use_kappa_manual, // ki
		  const double & kappa_manual_value, // ki	     
		  const bool & iso, // pii
		  const bool & correct_surface_emissions,
		  const bool & interp_and_calc_flux_step
		  
)
{
  dim3 calc_surf_grid(int((nbin + 15)/16), 1 , 1);  
  dim3 calc_surf_blocks(16, 1, 1);
  // csp
  calc_surface_planck<<<calc_surf_grid, calc_surf_blocks>>>(
							    dev_planckband_lay,
							    dev_opac_interwave,
							    dev_opac_deltawave,
							    nbin,
							    nlayer,
							    T_surf
							    );

  cudaDeviceSynchronize();
 
 if (correct_surface_emissions)
   {
     dim3 corr_surf_emiss_grid(int((nbin + 15)/16), 1 , 1);
     dim3 corr_surf_emiss_block(16,1,1);
     // cse
     correct_surface_emission<<<corr_surf_emiss_grid,
       corr_surf_emiss_block>>>(dev_F_down_tot,
				dev_opac_deltawave,
				dev_planckband_lay,
				surf_albedo,
				T_surf,
				nbin,
				nlayer,
				iter_value
				);

       cudaDeviceSynchronize();
   }

 // it
 dim3 it_grid(int( (ninterface+15)/16), 1, 1);
 dim3 it_block(16,1,1);
   
 interpolate_temperature<<<it_grid, it_block>>>(dev_T_lay,
			 dev_T_int,
			 ninterface);
   cudaDeviceSynchronize();

   // pil
   dim3 pil_grid(int((nbin+15)/16),int(((nlayer+2)+15))/16, 1);
   dim3 pil_block(16,16,1);
   planck_interpol_layer<<<pil_grid, pil_block>>>(dev_T_lay,
                         dev_planckband_lay,
                         dev_planckband_grid,
                         dev_starflux,
                         real_star,
                         nlayer,
                         nbin,
			 T_surf,
			 plancktable_dim,
			 plancktable_step
                        );
   cudaDeviceSynchronize();

   if (!iso)
     {
       // pii
       dim3 pii_grid(int((nbin+15)/16), int((ninterface+15)/16), 1 );
       dim3 pii_block(16,16,1);
       planck_interpol_interface<<<pii_grid,pii_block>>>(dev_T_int,
							 dev_planckband_int,
							 dev_planckband_grid,
							 ninterface,
							 nbin,
							 plancktable_dim,
							 plancktable_step);
       cudaDeviceSynchronize();
     }

   if (interp_and_calc_flux_step)
     {
       // io
       dim3 io_grid(int((nbin+15)/16), int((nlayer+15)/16), 1);
       dim3 io_block(16,16,1);

       interpolate_opacities<<<io_grid, io_block>>>(dev_T_lay,
						    dev_ktemp,
						    dev_p_lay,
						    dev_kpress,
						    dev_opac_k,
						    dev_opac_wg_lay,
						    dev_opac_scat_cross,
						    dev_scat_cross_lay,
						    npress,
						    ntemp,
						    ny,
						    nbin,
						    fake_opac,
						    nlayer);
						    
       
       cudaDeviceSynchronize();

       if (!iso)
	 {
	   // ioi
	   dim3 ioi_grid(int((nbin+15)/16), int((ninterface+15)/16), 1);
	   dim3 ioi_block(16,16,1);
	   
	   interpolate_opacities<<<ioi_grid, ioi_block>>>(dev_T_int,
							dev_ktemp,
							dev_p_int,
							dev_kpress,
							dev_opac_k,
							dev_opac_wg_int,
							dev_opac_scat_cross,
							dev_scat_cross_int,
							npress,
							ntemp,
							ny,
							nbin,
							fake_opac,
							ninterface);
	   
	   cudaDeviceSynchronize();
	 }

       // mmm
       dim3 mmm_block(16,1,1);
       dim3 mmm_grid(int((nlayer + 15)/16), 1, 1);
       
       meanmolmass_interpol<<<mmm_grid, mmm_block>>>(dev_T_lay,
					     dev_ktemp,
					     dev_meanmolmass_lay,
					     dev_opac_meanmass,
					     dev_p_lay,
					     dev_kpress,
					     npress,
					     ntemp,
					     nlayer);
			
	   
       cudaDeviceSynchronize();

       if (!iso)
	 {
	   // mmmi
	   dim3 mmmi_block(16,1,1);
	   dim3 mmmi_grid(int((ninterface + 15)/16), 1, 1);
	   
	   meanmolmass_interpol<<<mmmi_grid, mmmi_block>>>(dev_T_int,
						   dev_ktemp,
						   dev_meanmolmass_int,
						   dev_opac_meanmass,
						   dev_p_int,
						   dev_kpress,
						   npress,
						   ntemp,
						   ninterface);
	   
	   
	   cudaDeviceSynchronize();
	   
	 }

       // kappa interpolation
       double kappa_kernel_value = 0;
       if (use_kappa_manual)
	 kappa_kernel_value = kappa_manual_value;

       // kil
       dim3 kil_grid(int((nlayer + 15)/16), 1, 1);
       dim3 kil_block(16,1,1);
       
       kappa_interpol<<<kil_grid, kil_block>>>(dev_T_lay,
					     dev_entr_temp,
					     dev_p_lay,
					     dev_entr_press,
					     dev_kappa_lay,
					     dev_opac_kappa,
					     entr_npress,
					     entr_ntemp,
					     nlayer,
					     kappa_kernel_value);

       cudaDeviceSynchronize();

       if (!iso)
	 {
	   // kii
	   dim3 kii_grid(int((ninterface + 15)/16), 1, 1);
	   dim3 kii_block(16,1,1);
	   
	   kappa_interpol<<<kii_grid, kii_block>>>(dev_T_int,
						   dev_entr_temp,
						   dev_p_int,
						   dev_entr_press,
						   dev_kappa_int,
						   dev_opac_kappa,
						   entr_npress,
						   entr_ntemp,
						   ninterface,
						   kappa_kernel_value);

	   cudaDeviceSynchronize();
	 }
     }

   // TODO: add state check and return value
   
   return true;
}

bool calculate_transmission()
{

  return true;
}


void wrap_integrate_flux(long deltalambda_, // double*
			 long F_down_tot_, // double *
			 long F_up_tot_, // double *
			 long F_net_,  // double *
			 long F_down_wg_,  // double *
			 long F_up_wg_, // double *
			 long F_dir_wg_, // double *
			 long F_down_band_,  // double *
			 long F_up_band_,  // double *
			 long F_dir_band_, // double *
			 long gauss_weight_, // double *
			 int 	nbin, 
			 int 	numinterfaces, 
			 int 	ny,
			 int block_x,
			 int block_y,
			 int block_z,
			 int grid_x,
			 int grid_y,
			 int grid_z
			 )
{
  double* deltalambda = (double*)deltalambda_;
  double* F_down_tot = (double*)F_down_tot_;
  double* F_up_tot = (double*)F_up_tot_;
  double* F_net = (double*) F_net_;
  double* F_down_wg = (double*)F_down_wg_;
  double* F_up_wg = (double*)F_up_wg_;
  double* F_dir_wg = (double*)F_dir_wg_;
  double* F_down_band = (double*) F_down_band_;
  double* F_up_band = (double*) F_up_band_;
  double* F_dir_band = (double*)F_dir_band_;
  double* gauss_weight = (double*)gauss_weight_;

  dim3 threadsPerBlock(grid_x, grid_y, grid_z);
  dim3 numBlocks(block_x, block_y, block_z);

  printf("Running Alfrodull Wrapper for integrate flux\n");
  integrate_flux_double<<<threadsPerBlock, numBlocks>>>(
							deltalambda, 
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
}
