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
