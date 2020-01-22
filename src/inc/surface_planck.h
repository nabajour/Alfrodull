


// calculates the planck function for given surface temperature
__global__ void calc_surface_planck(
				    double* planckband_lay, 
				    double* lambda_edge, 
				    double* deltalambda,
				    int 	nwave,
				    int     numlayers,
				    double 	T_surf
				    );
