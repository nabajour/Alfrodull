#include "cuda_device_memory.h"
#include "gauss_legendre_weights.h"
#include "integrate_flux.h"

#include <algorithm> // std::max
#include <cstdio>
#include <memory>

using std::max;


bool cmp_dbl(double d1, double d2, double eps) {
  if (d1 == d2)
    return true;
  else if (d1 == 0.0 && d2 == 0.0)
    return false;

  return fabs(d1 - d2) / fabs(max(d1, d2)) < eps;
}

int main(int argc, char** argv) {
    bool success = false;
    printf("Running Flux Integration test\n");


    int point_num = 1;

    int nlayer = 15;

    int nbin = 10;
    int ny   = 5;

    int ninterface         = nlayer + 1;
    int nlayer_nbin        = nlayer * nbin;
    int ninterface_nbin    = ninterface * nbin;
    int ninterface_wg_nbin = ninterface * ny * nbin;

    printf("nlayer: %d\n", nlayer);
    printf("ninterface: %d\n", ninterface);
    printf("nbin: %d\n", nbin);
    printf("ny: %d\n", ny);
    // initialise arrays

    cuda_device_memory<double> F_down_wg;
    cuda_device_memory<double> F_up_wg;
    cuda_device_memory<double> F_dir_wg;

    cuda_device_memory<double> F_down_band;
    cuda_device_memory<double> F_up_band;
    cuda_device_memory<double> F_up_TOA_band;
    cuda_device_memory<double> F_dir_band;

    cuda_device_memory<double> F_down_band_opt;
    cuda_device_memory<double> F_up_band_opt;
    cuda_device_memory<double> F_dir_band_opt;

    cuda_device_memory<double> gauss_weights;
    cuda_device_memory<double> deltalambda;

    cuda_device_memory<double> F_down_tot;
    cuda_device_memory<double> F_up_tot;
    cuda_device_memory<double> F_dir_tot;
    cuda_device_memory<double> F_net;

    cuda_device_memory<double> F_down_tot_opt;
    cuda_device_memory<double> F_up_tot_opt;
    cuda_device_memory<double> F_net_opt;

    gauss_weights.allocate(ny);

    std::unique_ptr<double[]> weights = std::make_unique<double[]>(100);
    for (int i = 0; i < ny; i++)
        weights[i] = gauss_legendre_weights[ny - 1][i];

    gauss_weights.put(weights);


    deltalambda.allocate(point_num * nbin);

    F_down_wg.allocate(point_num * ninterface_wg_nbin);
    F_up_wg.allocate(point_num * ninterface_wg_nbin);
    F_dir_wg.allocate(point_num * ninterface_wg_nbin);

    F_down_band.allocate(point_num * ninterface_nbin);
    F_up_band.allocate(point_num * ninterface_nbin);
    F_dir_band.allocate(point_num * ninterface_nbin);

    F_up_TOA_band.allocate(point_num * nbin);
    
    F_down_tot.allocate(point_num * ninterface);
    F_up_tot.allocate(point_num * ninterface);
    F_dir_tot.allocate(point_num * ninterface);
    F_net.allocate(point_num * ninterface);

    F_down_band_opt.allocate(point_num * ninterface_nbin);
    F_up_band_opt.allocate(point_num * ninterface_nbin);
    F_dir_band_opt.allocate(point_num * ninterface_nbin);

    F_down_tot_opt.allocate(point_num * ninterface);
    F_up_tot_opt.allocate(point_num * ninterface);
    F_net_opt.allocate(point_num * ninterface);


    // initialise data
    // get datat pointers
    std::shared_ptr<double[]> deltalambda_h = deltalambda.get_host_data_ptr();
    std::shared_ptr<double[]> F_up_wg_h     = F_up_wg.get_host_data_ptr();
    std::shared_ptr<double[]> F_down_wg_h   = F_down_wg.get_host_data_ptr();
    std::shared_ptr<double[]> F_dir_wg_h    = F_dir_wg.get_host_data_ptr();


    // fill data
    for (int i = 0; i < nbin; i++) {
      deltalambda_h[i] = 1.0;
    }

    for (int i = 0; i < ninterface; i++) {
        for (int j = 0; j < nbin; j++) {
            for (int k = 0; k < ny; k++) {
                //printf("%d %d %d \n", i, j, k);
	      F_down_wg_h[i * nbin * ny + j * ny + k] = 1.0/double(i+1 + j + k + 0.5);
                F_up_wg_h[i * nbin * ny + j * ny + k]   = 1.0/double(i+1 + j + k + 0.9);
                F_dir_wg_h[i * nbin * ny + j * ny + k]  = 1.0/double(i+1 + j + k + 0.7);
            }
        }
    }

    deltalambda.put();
    F_down_wg.put();
    F_up_wg.put();
    F_dir_wg.put();

    // compute
    printf("Original version\n");
    // optimised version
    {
        int  num_levels_per_block = 256 / nbin + 1;
        dim3 gridsize(ninterface / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block, nbin);
        //printf("nbin: %d, ny: %d\n", nbin, ny);

        integrate_flux_band<<<gridsize, blocksize>>>(*F_down_wg,
                                                     *F_up_wg,
                                                     *F_dir_wg,
                                                     *F_down_band_opt,
                                                     *F_up_band_opt,
                                                     *F_dir_band_opt,
                                                     *F_up_TOA_band,
                                                     *gauss_weights,
                                                     nbin,
                                                     ninterface,
                                                     ny);

        cudaDeviceSynchronize();
    }

    {
        int  num_levels_per_block = 256 ;
        dim3 gridsize(ninterface / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block);
        integrate_flux_tot<<<gridsize, blocksize>>>(*deltalambda,
                                                    *F_down_tot_opt,
                                                    *F_up_tot_opt,
                                                    *F_dir_tot,
                                                    *F_net_opt,
                                                    *F_down_band_opt,
                                                    *F_up_band_opt,
                                                    *F_dir_band_opt,
                                                    nbin,
                                                    ninterface);

        cudaDeviceSynchronize();
    }


    // original version
    printf("Original test\n");
    {

        dim3 threadsPerBlock(1, 1, 1);
        dim3 numBlocks(32, 4, 8);


        //printf("Running Alfrodull Wrapper for integrate flux\n");
        integrate_flux_double<<<threadsPerBlock, numBlocks>>>(*deltalambda,
                                                              *F_down_tot,
                                                              *F_up_tot,
                                                              *F_net,
                                                              *F_down_wg,
                                                              *F_up_wg,
                                                              *F_dir_wg,
                                                              *F_down_band,
                                                              *F_up_band,
                                                              *F_dir_band,
                                                              *gauss_weights,
                                                              nbin,
                                                              ninterface,
                                                              ny);

        cudaDeviceSynchronize();
    }
    printf("Comparing output\n");
    printf("Comparing band\n");

    bool debug = true;

    int error = 0;
    int total = 0;

    std::shared_ptr<double[]> F_up_band_h   = F_up_band.get_host_data();
    std::shared_ptr<double[]> F_down_band_h = F_down_band.get_host_data();
    std::shared_ptr<double[]> F_dir_band_h  = F_dir_band.get_host_data();

    std::shared_ptr<double[]> F_up_band_opt_h   = F_up_band_opt.get_host_data();
    std::shared_ptr<double[]> F_down_band_opt_h = F_down_band_opt.get_host_data();
    std::shared_ptr<double[]> F_dir_band_opt_h  = F_dir_band_opt.get_host_data();


    double epsilon = 1e-12;
    for (int i = 0; i < ninterface; i++) {
        for (int j = 0; j < nbin; j++) {
            double f_dwn_bd     = F_down_band_h[i * nbin + j];
            double f_dwn_bd_opt = F_down_band_opt_h[i * nbin + j];

            bool match1 = cmp_dbl(f_dwn_bd, f_dwn_bd_opt, epsilon);

            double f_up_bd     = F_up_band_h[i * nbin + j];
            double f_up_bd_opt = F_up_band_opt_h[i * nbin + j];

            bool match2 = cmp_dbl(f_up_bd, f_up_bd_opt, epsilon);

            double f_dir_bd     = F_dir_band_h[i * nbin + j];
            double f_dir_bd_opt = F_dir_band_opt_h[i * nbin + j];

            bool match3 = cmp_dbl(f_dir_bd, f_dir_bd_opt, epsilon);

            if (match1 && match2 && match3) {
                if (debug)
		  printf("% 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g %5d\n",
                           f_dwn_bd_opt,
                           f_dwn_bd,
                           match1,
                           f_up_bd_opt,
                           f_up_bd,
                           match2,
                           f_dir_bd_opt,
                           f_dir_bd,
                           match3);
            }
            else {
                error += 1;

                printf("% 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g %5d\n",
                       f_dwn_bd_opt,
                       f_dwn_bd,
                       match1,
                       f_up_bd_opt,
                       f_up_bd,
                       match2,
                       f_dir_bd_opt,
                       f_dir_bd,
                       match3);
            }
            total += 1;
        }
    }
    printf("errors: %d/%d\n", error, total);
    
    printf("Comparing total\n");

    std::shared_ptr<double[]> F_up_tot_h   = F_up_tot.get_host_data();
    std::shared_ptr<double[]> F_down_tot_h = F_down_tot.get_host_data();
    std::shared_ptr<double[]> F_net_h      = F_net.get_host_data();

    std::shared_ptr<double[]> F_up_tot_opt_h   = F_up_tot_opt.get_host_data();
    std::shared_ptr<double[]> F_down_tot_opt_h = F_down_tot_opt.get_host_data();
    std::shared_ptr<double[]> F_net_opt_h      = F_net_opt.get_host_data();

    for (int i = 0; i < ninterface; i++) {
      double f_dwn_tot     = F_down_tot_h[i];
      double f_dwn_tot_opt = F_down_tot_opt_h[i];

      bool match1 = cmp_dbl(f_dwn_tot, f_dwn_tot_opt, epsilon);

      double f_up_tot     = F_up_tot_h[i];
      double f_up_tot_opt = F_up_tot_opt_h[i];
      
      bool match2 = cmp_dbl(f_up_tot, f_up_tot_opt, epsilon);

      double f_net     = F_net_h[i];
      double f_net_opt = F_net_opt_h[i];

      bool match3 = cmp_dbl(f_net, f_net_opt, epsilon);
      
      if (match1 && match2 && match3) {
	if (debug)
	  printf("% 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g %5d\n",
		 f_dwn_tot_opt,
		 f_dwn_tot,
		 match1,
		 f_up_tot_opt,
		 f_up_tot,
		 match2,
		 f_net_opt,
		 f_net,
		 match3);
      }
            else {
	      error += 1;
	      
	      printf("% 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g %5d\n",
		     f_dwn_tot_opt,
                       f_dwn_tot,
		     match1,
		     f_up_tot_opt,
		     f_up_tot,
		     match2,
                       f_net_opt,
		     f_net,
		     match3);
            }
            total += 1;
    
    }


    printf("errors: %d/%d\n", error, total);

    printf("Flux Integration test done\n");


    if (success) {
        printf("Success\n");
        return 0;
    }
    else {
        printf("Fail\n");
        return -1;
    }
}
