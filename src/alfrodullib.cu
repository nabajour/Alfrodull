#include "inc/integrate_flux.h"


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
