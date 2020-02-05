#include "opacities.h"
#include "cudaDeviceMemory.h"
#include "planck_table.h"


class alfrodull_engine
{
public:
  alfrodull_engine();

  void init();
  
  void load_opacities(const string & filename);
  void prepare_planck_table();

  void set_parameters(const int & nlayer_,
		      const bool & iso_,
		      const double & T_star_);

  void allocate_internal_variables();

  // TODO: temporary prototyping wrapper for HELIOS. 
  void get_device_pointers_for_helios_write(double *& dev_scat_cross_section_lay,
					    double *& dev_scat_cross_section_int,
					    double *& dev_interwave,
					    double *& dev_deltawave,
					    double *& dev_planck_lay,
					    double *& dev_planck_int,
					    double *& dev_planck_grid,
					    double *& dev_delta_tau_wg,
					    double *& dev_delta_tau_wg_upper,
					    double *& dev_delta_tau_wg_lower,
					    int & dim,
					    int & step);

  void correct_incident_energy(double * starflux_array_ptr,
			       bool real_star,
			       bool energy_budge_correction);
  //private:
  opacity_table opacities;

  planck_table plancktable;

  // general sim parameters
  int nbin = 0; // should come from opacity table (?)

  int nlayer = 0;
  int ninterface = 0; // nlayers + 1
  bool iso = false;
  double T_star = 0.0;
  
  // device memory
  //  scattering
  cuda_device_memory<double> scatter_cross_section_lay;
  cuda_device_memory<double> scatter_cross_section_inter;

  // planck function 
  cuda_device_memory<double> planckband_lay;
  cuda_device_memory<double> planckband_int;

  // delta tau, for weights. Only used internally (on device) for flux computations
  // and shared at the end for integration over wg
  // iso
  cuda_device_memory<double> delta_tau_wg;
  // noiso
  cuda_device_memory<double> delta_tau_wg_upper;
  cuda_device_memory<double> delta_tau_wg_lower;
  
  // Flux computation quantities
  // computed in trans_iso/trans_noniso
  // used in populate_spectral_flux (iso/non_iso)
  // iso
  cuda_device_memory<double> M_term;
  cuda_device_memory<double> N_term;   
  cuda_device_memory<double> P_term;
  cuda_device_memory<double> G_plus;
  cuda_device_memory<double> G_minus;
  cuda_device_memory<double> w_0;
  // noniso
  cuda_device_memory<double> M_upper;
  cuda_device_memory<double> M_lower;
  cuda_device_memory<double> N_upper;
  cuda_device_memory<double> N_lower;
  cuda_device_memory<double> P_upper;
  cuda_device_memory<double> P_lower;
  cuda_device_memory<double> G_plus_upper;
  cuda_device_memory<double> G_plus_lower;
  cuda_device_memory<double> G_minus_upper;
  cuda_device_memory<double> G_minus_lower;
  cuda_device_memory<double> w_0_upper;
  cuda_device_memory<double> w_0_lower;
};
