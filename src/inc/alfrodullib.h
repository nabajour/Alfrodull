

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
			  const int & entr_npress, // kii, kil
			  const int & entr_ntemp, // kii, kil	  
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
			  
			       );

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
		  
			  );


// calculates the integrated upwards and downwards fluxes
void integrate_flux(
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
		    );

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
			 );
