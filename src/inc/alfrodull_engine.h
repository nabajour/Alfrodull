#include "opacities.h"
#include "cudaDeviceMemory.h"

class alfrodull_engine
{
public:
  alfrodull_engine();

  void init();
  
  void load_opacities(const string & filename);

  void set_parameters(const int & nlayer_,
		      const bool & iso_);

  void allocate_internal_variables();

  // TODO: temporary prototyping wrapper for HELIOS. 
  void get_device_pointers_for_helios_write(double *& dev_scat_cross_section_lay,
					    double *& dev_scat_cross_section_int,
					    double *& dev_interwave,
					    double *& dev_deltawave);
  
  //private:
  opacity_table opacities;

  // general sim parameters
  int nbin = 0; // should come from opacity table (?)

  int nlayer = 0;
  int ninterface = 0; // nlayers + 1
  bool iso = false;
  
  // device memory
  cuda_device_memory<double> scatter_cross_section_lay;
  cuda_device_memory<double> scatter_cross_section_inter;

  // Flux computation quantities
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
