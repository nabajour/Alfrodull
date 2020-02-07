#include <tuple>
#include <functional>

void wrap_compute_radiative_transfer(
			 // prepare_compute_flux
				     long dev_starflux,              // in: pil
				     // state variables
				     long       dev_T_lay,           // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
				     long       dev_T_int,           // in: it, pii, ioi, mmmi, kii  
				     long       dev_p_lay,           // in: io, mmm, kil
				     long       dev_p_int,           // in: ioi, mmmi, kii
				     long       opac_wg_lay,     // out: io
				     long       opac_wg_int,     // out: ioi
				     long       meanmolmass_lay, // out: mmm
				     long       meanmolmass_int, // out: mmmi
				     const int&    ninterface,          // it, pii, mmmi, kii
				     const int&    real_star,        // pil
				     const double& fake_opac,        // io
				     const double& T_surf,           // csp, cse, pil
				     const double& surf_albedo,      // cse
				     const bool&   correct_surface_emissions,
				     const bool&   interp_and_calc_flux_step,
				     // calculate_transmission_iso
				     long trans_wg,        // out
				     
				     // calculate_transmission_non_iso
				     long trans_wg_upper,
				     long trans_wg_lower,
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
				     int     ny,
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
				     long deltalambda, // -> dev_opac_deltawave
				     long F_down_tot,
				     long F_up_tot,
				     long F_net,
				     long F_down_band,
				     long F_up_band,
				     long F_dir_band,
				     long gauss_weight	
);

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
			  const int & ninterface, // it, pii, mmmi, kii
			  const int & real_star, // pil
			  const double & fake_opac, // io
			  const double & T_surf, // csp, cse, pil
			  const double & surf_albedo, // cse
			  const bool & correct_surface_emissions,
			  const bool & interp_and_calc_flux_step
			       );

bool prepare_compute_flux(
		  double * dev_starflux, // pil
		  double * dev_T_lay, // it, pil, io, mmm, kil
		  double * dev_T_int, // it, pii, ioi, mmmi, kii
		  double * dev_p_lay, // io, mmm, kil
		  double * dev_p_int, // ioi, mmmi, kii
		  double * dev_opac_wg_lay, // io
		  double * dev_opac_wg_int, // ioi
		  double * dev_meanmolmass_lay, // mmm
		  double * dev_meanmolmass_int, // mmmi
		  const int & ninterface, // it, pii, mmmi, kii
		  const int & real_star, // pil
		  const double & fake_opac, // io
		  const double & T_surf, // csp, cse, pil
		  const double & surf_albedo, // cse
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
			 int 	numinterfaces, 
			 int 	ny
			 );

bool wrap_calculate_transmission_iso(
				      long 	trans_wg,
        long 	delta_colmass,
        long 	opac_wg_lay,
        long cloud_opac_lay,
        long 	meanmolmass_lay,
        long 	cloud_scat_cross_lay,
        long 	g_0_tot_lay,
        double   g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	ny,
        int 	clouds,
        int 	scat_corr
					      );

 bool wrap_calculate_transmission_noniso(
        long trans_wg_upper,
        long trans_wg_lower,
        long delta_col_upper,
        long delta_col_lower,
        long opac_wg_lay,
        long opac_wg_int,
        long cloud_opac_lay,
        long cloud_opac_int,		
        long meanmolmass_lay,
        long meanmolmass_int,
        long cloud_scat_cross_lay,
        long cloud_scat_cross_int,		
        long 	g_0_tot_lay,
        long 	g_0_tot_int,
        double	g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	ny,
        int 	clouds,
        int 	scat_corr
					 );


bool wrap_direct_beam_flux(long 	F_dir_wg,
			   long 	Fc_dir_wg,
			   long 	z_lay,
			   double 	mu_star,
			   double	R_planet,
			   double 	R_star, 
			   double 	a,
			   int		dir_beam,
			   int		geom_zenith_corr,
			   int 	ninterface,
			   int 	ny			   );




bool wrap_populate_spectral_flux_noniso(
					      long F_down_wg, 
					      long F_up_wg, 
					      long Fc_down_wg, 
					      long Fc_up_wg,
					      long F_dir_wg,
					      long Fc_dir_wg,
					      long g_0_tot_lay,
					      long g_0_tot_int,
					      double 	g_0,
					      int 	singlewalk, 
					      double 	Rstar, 
					      double 	a, 
					      int 	numinterfaces,
					      double 	f_factor,
					      double 	mu_star,
					      int 	ny,
					      double 	epsi,
					      double 	w_0_limit,
					      double 	delta_tau_limit,
					      int 	dir_beam,
					      int 	clouds,
					      double   albedo,
					      long	trans_wg_upper,
					      long trans_wg_lower
					);

bool wrap_populate_spectral_flux_iso(
				     long F_down_wg, 
        long F_up_wg, 
        long F_dir_wg, 
        long g_0_tot_lay,
        double 	g_0,
        int 	singlewalk, 
        double 	Rstar, 
        double 	a, 
        int 	numinterfaces, 
        double 	f_factor, 
        double 	mu_star,
        int 	ny, 
        double 	epsi,
        double 	w_0_limit,
        int 	dir_beam,
        int 	clouds,
        double   albedo
				     );

void init_alfrodull();
void init_parameters(const int & nlayer_,
		     const bool & iso_,
		     const double & Tstar_);
void deinit_alfrodull();

void set_z_calc_function(std::function<void()> & func);

// TODO: this shouldn't be visible externally
void allocate();

std::tuple<long, long, long, long, long, long, long, long, long, long, long, long, long, int, int> get_device_pointers_for_helios_write();

void prepare_planck_table();
void correct_incident_energy(long starflux_array_ptr,
			     bool real_star,
			     bool energy_budge_correction);


