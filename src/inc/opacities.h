#include <memory>
#include <string>


#include "cudaDeviceMemory.h"
#include "storage.h"

// class to handle opacities and meanmolmass from k_table
// loads data table
// initialise variables for ref table management
// prepares GPU data
// loads data to GPU
// interpolates data
class opacity_table
{
public:
  opacity_table();
  
  bool load_opacity_table(const string & filename);
  //private:
  string opacity_filename;


  // device tables
  cuda_device_memory<double> dev_kpoints;

  cuda_device_memory<double> dev_temperatures;
  int                        n_temperatures = 0;
  cuda_device_memory<double> dev_pressures;
  int                        n_pressures = 0;
  
  // wieghted Rayeigh c
  cuda_device_memory<double> dev_scat_cross_sections;
  
  // Mean molecular mass
  // TODO: needs to be in AMU
  cuda_device_memory<double> dev_meanmolmass;
  
  cuda_device_memory<double> dev_opac_wave;
  int                        nbin = 0;
  
  cuda_device_memory<double> dev_opac_y;
  int                        ny = 0;
  
  cuda_device_memory<double> dev_opac_interwave;
  
  cuda_device_memory<double> dev_opac_deltawave;
  
  // needed for interpolate_opacities
    // dev_T_lay
    // dev_ktemp
    // dev_p_lay,
    // 					      dev_kpress,
    // 						    dev_opac_k,
    // 						    dev_opac_wg_lay,
    // 						    dev_opac_scat_cross,
    // 						    dev_scat_cross_lay,
    // 						    npress,
    // 						    ntemp,
    // 						    ny,
    // 						    nbin,
    // 						    fake_opac,
    // 						    nlayer
};
