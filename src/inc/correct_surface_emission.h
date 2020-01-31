
// adjust the surface emission to satisfy the Stefan-Boltzmann law for the surface temperature
__global__ void correct_surface_emission(double*  delta_lambda,
					 double* 	planckband_lay,
					 double surf_albedo,
					 double T_surf,
					 int    nwave,
					 int     numlayers
					 );
