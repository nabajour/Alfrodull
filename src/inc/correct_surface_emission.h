
// adjust the surface emission to satisfy the Stefan-Boltzmann law for the surface temperature
__global__ void corr_surface_emission(
				      double*  F_down_tot,
				      double*  delta_lambda,
				      double* 	planckband_lay,
				      double surf_albedo,
				      double T_surf,
				      int    nwave,
				      int     numlayers,
				      int     itervalue
				      );
