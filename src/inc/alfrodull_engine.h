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
  
};
