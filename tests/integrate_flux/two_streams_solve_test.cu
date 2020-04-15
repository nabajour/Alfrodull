#include "cuda_device_memory.h"
#include "gauss_legendre_weights.h"
#include "integrate_flux.h"

#include <algorithm> // std::max
#include <cstdio>
#include <memory>
#include <random>

#include <chrono>
#include <iomanip>
#include <sstream>


using std::max;


void cuda_check_status_or_exit(const char* filename, const int& line) {
    cudaError_t err = cudaGetLastError();

    // Check device query
    if (err != cudaSuccess) {
        printf("[%s:%d] CUDA error check reports error: %s\n",
                    filename,
                    line,
                    cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

bool cmp_dbl(double d1, double d2, double eps) {
  if (d1 == d2)
    return true;
  else if (d1 == 0.0 && d2 == 0.0)
    return false;

  return fabs(d1 - d2) / fabs(max(d1, d2)) < eps;
}

int main(int argc, char** argv) {
    bool success = true;
    printf("Running Two Streams Solver test\n");


    int point_num = 1;

    int nlayer = 15;

    int nbin = 1;
    int ny   = 1;

    int ninterface         = nlayer + 1;
    int nlayer_nbin        = nlayer * nbin;
    int nlayer_plus2_nbin  = (nlayer + 2) * nbin;
    int nlayer_wg_nbin     = nlayer * nbin * ny;
    int ninterface_nbin    = ninterface * nbin;
    int ninterface_wg_nbin = ninterface * ny * nbin;

    printf("nlayer: %d\n", nlayer);
    printf("ninterface: %d\n", ninterface);
    printf("nbin: %d\n", nbin);
    printf("ny: %d\n", ny);
    // initialise arrays

    bool iso = true;
	
    cuda_device_memory<double> F_down_wg_helios;
    cuda_device_memory<double> F_up_wg_helios;

    cuda_device_memory<double> F_down_wg_alf;
    cuda_device_memory<double> F_up_wg_alf;

    cuda_device_memory<double> F_dir_wg;

    cuda_device_memory<double> planckband_lay;
    cuda_device_memory<double> w_0;
    cuda_device_memory<double> M_term;
    cuda_device_memory<double> N_term;
    cuda_device_memory<double> P_term;
    cuda_device_memory<double> G_plus;
    cuda_device_memory<double> G_minus;
    cuda_device_memory<double> g_0_tot_lay;

    double g_0 = 0.5;
    bool single_walk = true;
    double Rstar = 1.0;
    double a = 1.0;
    double f_factor = 1.0;
    double mu_star = -1.0;
    double epsi = 0.5;
    bool dir_beam = false;
    bool clouds = false;
    bool scat_corr = true;
    double albedo = 1.0;
    double i2s_transition = 0.1;

    bool debug = true;

    F_down_wg_helios.allocate(point_num * ninterface_wg_nbin);
    F_up_wg_helios.allocate(point_num * ninterface_wg_nbin);

    F_down_wg_alf.allocate(point_num * ninterface_wg_nbin);
    F_up_wg_alf.allocate(point_num * ninterface_wg_nbin);

    F_dir_wg.allocate(point_num * ninterface_wg_nbin);

    F_down_wg_helios.zero();
    F_up_wg_helios.zero();

    F_down_wg_alf.zero();
    F_up_wg_alf.zero();
    
    
    planckband_lay.allocate(nlayer_plus2_nbin);

    // Random number generator
    std::random_device rd;        //Will be used to obtain a seed for the random number engine
    std::mt19937       gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> rand_norm_dist{1.0,0.5};

    std::normal_distribution<> rand_norm_dist_plk{1e-5,0.5};
	
    if (iso) {
      printf("Initialising data for ISO case\n");
        M_term.allocate(nlayer_wg_nbin);
        N_term.allocate(nlayer_wg_nbin);
        P_term.allocate(nlayer_wg_nbin);
        G_plus.allocate(nlayer_wg_nbin);
        G_minus.allocate(nlayer_wg_nbin);
        w_0.allocate(nlayer_wg_nbin);
	g_0_tot_lay.allocate(nlayer_nbin);

	// initialise data
	// get data pointers
	std::shared_ptr<double[]> M_term_h      = M_term.get_host_data_ptr();
	std::shared_ptr<double[]> N_term_h      = N_term.get_host_data_ptr();
	std::shared_ptr<double[]> P_term_h      = P_term.get_host_data_ptr();
	std::shared_ptr<double[]> G_plus_h      = G_plus.get_host_data_ptr();
	std::shared_ptr<double[]> G_minus_h     = G_minus.get_host_data_ptr();
	std::shared_ptr<double[]> w_0_h         = w_0.get_host_data_ptr();
	std::shared_ptr<double[]> g_0_tot_lay_h = g_0_tot_lay.get_host_data_ptr();
	std::shared_ptr<double[]> planckband_lay_h = planckband_lay.get_host_data_ptr();
	       
	for (int i = 0; i < nlayer; i++) {
	  for (int j = 0; j < nbin; j++) {
	    for (int k = 0; k < ny; k++) {
	      // printf("%d %d %d \n", i, j, k);
	      double T = fabs(rand_norm_dist(gen));
	      double z_p = fabs(rand_norm_dist(gen));
	      double z_m = fabs(rand_norm_dist(gen));

	      M_term_h[k + ny*j + i*ny*nbin] = z_m*z_m*T*T - z_p*z_p;

	      double n_term = 0.0;
	      if (scat_corr)
		n_term = z_m*z_p*(1.0 - T*T);
	      
	      N_term_h[k + ny*j + i*ny*nbin] = n_term;
	      
	      P_term_h[k + ny*j + i*ny*nbin] = (z_m*z_m - z_p*z_p)*T;
	      
	      double a = fabs(rand_norm_dist(gen));
	      double b = fabs(rand_norm_dist(gen));
	      G_plus_h[k + ny*j + i*ny*nbin] = max(a, b);
	      G_minus_h[k + ny*j + i*ny*nbin] = min(a,b);
	      w_0_h[k + ny*j + i*ny*nbin] = 0.5; //dis(gen);
	      // printf("%g %g %g %g %g %g %g\n",
	      // 	     M_term_h[k + ny*j + i*ny*nbin],
	      // 	     N_term_h[k + ny*j + i*ny*nbin],
	      // 	     P_term_h[k + ny*j + i*ny*nbin],
	      // 	     G_plus_h[k + ny*j + i*ny*nbin],
	      // 	     G_minus_h[k + ny*j + i*ny*nbin],
	      // 	     w_0_h[k + ny*j + i*ny*nbin]);
	    }
	    g_0_tot_lay_h[j + i*nbin] = rand_norm_dist(gen);
	  }
	}

	for (int i = 0; i < nlayer; i++) {
	  for (int j = 0; j < nbin; j++) {
	    planckband_lay_h[j*(ninterface -1+2) + i] = 0.0;
	    //planckband_lay_h[j*(ninterface -1+2) + i] = fabs(rand_norm_dist(gen));
	  }
	}
	for (int j = 0; j < nbin; j++) {
	  planckband_lay_h[j*(ninterface -1+2) + nlayer ] = fabs(rand_norm_dist_plk(gen));
	  planckband_lay_h[j*(ninterface -1+2) + nlayer + 1] = fabs(rand_norm_dist_plk(gen));
	}
	
	M_term.put();
	N_term.put();
	P_term.put();
	G_plus.put();
	G_minus.put();
	w_0.put();
	g_0_tot_lay.put();
	planckband_lay.put();

	
    }
    else {
        // M_upper.allocate(nlayer_wg_nbin);
        // M_lower.allocate(nlayer_wg_nbin);
        // N_upper.allocate(nlayer_wg_nbin);
        // N_lower.allocate(nlayer_wg_nbin);
        // P_upper.allocate(nlayer_wg_nbin);
        // P_lower.allocate(nlayer_wg_nbin);
        // G_plus_upper.allocate(nlayer_wg_nbin);
        // G_plus_lower.allocate(nlayer_wg_nbin);
        // G_minus_upper.allocate(nlayer_wg_nbin);
        // G_minus_lower.allocate(nlayer_wg_nbin);
        // w_0_upper.allocate(nlayer_wg_nbin);
        // w_0_lower.allocate(nlayer_wg_nbin);
    }    

    {
      std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
          
      // compute
      printf("Original version\n");
      
      bool debug_fn = false;
      
      int nscat_step = 0;
      if (single_walk)
        nscat_step = 200;
      else
        nscat_step = 3;
      
      for (int scat_iter = 0; scat_iter < nscat_step * scat_corr + 1; scat_iter++) {
        if (iso) {
	  printf("Loop\n");
	  dim3 block(16, 16, 1);
	  dim3 grid((nbin + 15) / 16, (ny + 15) / 16, 1);
	  fband_iso_notabu<<<grid, block>>>(*F_down_wg_helios,
					    *F_up_wg_helios,
					    *F_dir_wg,
					    *planckband_lay,
					    *w_0,
					    *M_term,
					    *N_term,
					    *P_term,
					    *G_plus,
					    *G_minus,
					    *g_0_tot_lay,
					    g_0,
					    single_walk,
					    Rstar,
					    a,
					    ninterface,
					    nbin,
					    f_factor,
					    mu_star,
					    ny,
					    epsi,
					    dir_beam,
					    clouds,
					    scat_corr,
					    albedo,
					    debug_fn,
					    i2s_transition);
	  
	  cudaDeviceSynchronize();
	  cuda_check_status_or_exit(__FILE__, __LINE__);
        }
        else {
	  /*
	    int nbin = opacities.nbin;
	    int ny   = opacities.ny;
	    
	    dim3 block(16, 16, 1);
	    
	    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, 1);
	    
	  // calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
	  fband_noniso_notabu<<<grid, block>>>(F_down_wg,
	  F_up_wg,
					       Fc_down_wg,
					       Fc_up_wg,
					       F_dir_wg,
					       Fc_dir_wg,
					       *planckband_lay,
					       *planckband_int,
					       *w_0_upper,
					       *w_0_lower,
					       *delta_tau_wg_upper,
					       *delta_tau_wg_lower,
					       *M_upper,
					       *M_lower,
					       *N_upper,
					       *N_lower,
					       *P_upper,
					       *P_lower,
					       *G_plus_upper,
					       *G_plus_lower,
					       *G_minus_upper,
					       *G_minus_lower,
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
					       delta_tau_limit,
					       dir_beam,
					       clouds,
					       scat_corr,
					       albedo,
					       debug,
					       i2s_transition);
	  */
        }
      }

      std::chrono::system_clock::time_point stop  = std::chrono::system_clock::now();
      auto duration_helios = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

      printf("Computed in: HELIOS: %ld us\n", duration_helios.count());
      
    }
    cuda_check_status_or_exit(__FILE__, __LINE__);

    
    printf("Comparing output\n");
    

    int error = 0;
    int total = 0;

    std::shared_ptr<double[]> F_up_wg_helios_h   = F_up_wg_helios.get_host_data();
    std::shared_ptr<double[]> F_down_wg_helios_h   = F_down_wg_helios.get_host_data();
    std::shared_ptr<double[]> F_up_wg_alf_h   = F_up_wg_alf.get_host_data();
    std::shared_ptr<double[]> F_down_wg_alf_h   = F_down_wg_alf.get_host_data();
    
    double epsilon = 1e-12;
    for (int i = 0; i < ninterface; i++) {
      for (int j = 0; j < nbin; j++) {
	for (int k = 0; k < ny; k++) {
	  double f_dwn_wg_helios     = F_down_wg_helios_h[i * nbin * ny + j * ny + k];
	  double f_up_wg_helios     = F_up_wg_helios_h[i * nbin * ny + j * ny + k];

	  double f_dwn_wg_alf     = F_down_wg_alf_h[i * nbin * ny + j * ny + k];
	  double f_up_wg_alf     = F_up_wg_alf_h[i * nbin * ny + j * ny + k];
	  

	  bool match_dwn = cmp_dbl(f_dwn_wg_helios, f_dwn_wg_alf, epsilon);
	  bool match_up = cmp_dbl(f_up_wg_helios, f_up_wg_alf, epsilon);
	  
	  if (match_up && match_dwn) {
	    if (debug)
	      printf("% 5d % 5d % 5d % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d\n",
		     i,j,k,
		     f_dwn_wg_helios,
		     f_dwn_wg_alf,
		     match_dwn,
		     f_up_wg_helios,
		     f_up_wg_alf,
		     match_up);
            }
            else {
                error += 1;
		printf("% 5d % 5d % 5d % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d\n",
		       i,j,k,
		       f_dwn_wg_helios,
		       f_dwn_wg_alf,
		       match_dwn,
		       f_up_wg_helios,
		       f_up_wg_alf,
		       match_up);
            }
            total += 1;
	    success &= match_up && match_dwn;
        }
      }
    }
    printf("errors: %d/%d\n", error, total);
    
    printf("Two stream solver test done\n");


    if (success) {
        printf("Success\n");
        return 0;
    }
    else {
        printf("Fail\n");
        return -1;
    }
}
