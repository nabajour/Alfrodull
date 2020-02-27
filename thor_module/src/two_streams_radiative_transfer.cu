// ==============================================================================
// This file is part of THOR.
//
//     THOR is free software : you can redistribute it and / or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     THOR is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     THOR directory under <license.txt>.If not, see
//     <http://www.gnu.org/licenses/>.
// ==============================================================================
//
// Two stream radiative transfer
//
//
//
// Method: Helios Two Stream algorithm
//
//
// Known limitations: - Runs in a single GPU.
//
// Known issues: None
//
//
// If you use this code please cite the following reference:
//
//       [1] Mendonca, J.M., Grimm, S.L., Grosheintz, L., & Heng, K., ApJ, 829, 115, 2016
//
// Current Code Owner: Joao Mendonca, EEG. joao.mendonca@csh.unibe.ch
//
// History:
// Version Date       Comment
// ======= ====       =======
//
//
//
////////////////////////////////////////////////////////////////////////

#include "two_streams_radiative_transfer.h"

#include "alfrodullib.h"

two_streams_radiative_transfer::two_streams_radiative_transfer() {
}

two_streams_radiative_transfer::~two_streams_radiative_transfer() {
}

void two_streams_radiative_transfer::print_config() {
  printf("Alf_Tstar: %f\n", T_star );
  printf("Alf_iso: %s\n", iso?"true":"false" );
  printf("Alf_real_star: %s\n", real_star?"true":"false" );
  printf("Alf_fake_opac: %f\n", fake_opac );

  printf("Alf_T_surf: %f\n", T_surf );
  printf("Alf_albedo: %f\n", albedo );
  printf("Alf_g_0: %f\n", g_0 );
  printf("Alf_diffusivity: %d\n", epsi );

  printf("Alf_scat: %s\n", scat?"true":"false" );
  printf("Alf_scat_corr: %s\n", scat_corr?"true":"false" );
  printf("Alf_R_planet: %f\n", R_planet );
  printf("Alf_a: %f\n", a );
  
  printf("Alf_dir_beam: %s\n", dir_beam?"true":"false" );
  printf("Alf_geom_zenith_corr: %s\n", geom_zenith_corr?"true":"false" );

  printf("Alf_f_factor: %f\n", f_factor );
  printf("Alf_w_0_limit: %f\n", w_0_limit );
  printf("Alf_i2s_transition: %f\n", i2s_transition );
  printf("Alf_opacities_file: %s\n", opacities_file.c_str() );
}

bool two_streams_radiative_transfer::configure(config_file& config_reader) {
    config_reader.append_config_var("Alf_Tstar", T_star, T_star);
    config_reader.append_config_var("Alf_iso", iso, iso);
    config_reader.append_config_var("Alf_real_star", real_star, real_star);
    config_reader.append_config_var("Alf_fake_opac", fake_opac, fake_opac);
    config_reader.append_config_var("Alf_T_surf", T_surf, T_surf); // ?
    config_reader.append_config_var("Alf_albedo", albedo, albedo);
    config_reader.append_config_var("Alf_g_0", g_0, g_0);
    config_reader.append_config_var("Alf_diffusivity", epsi, epsi);

    config_reader.append_config_var("Alf_scat", scat, scat);
    config_reader.append_config_var("Alf_scat_corr", scat_corr, scat_corr);
    config_reader.append_config_var("Alf_R_planet", R_planet, R_planet);
    config_reader.append_config_var("Alf_a", a, a);
    config_reader.append_config_var("Alf_dir_beam", dir_beam, dir_beam);
    config_reader.append_config_var("Alf_geom_zenith_corr", geom_zenith_corr, geom_zenith_corr);
    config_reader.append_config_var("Alf_f_factor", f_factor, f_factor);
    config_reader.append_config_var("Alf_i2s_transition", i2s_transition, i2s_transition);

    config_reader.append_config_var("Alf_opacities_file", opacities_file, opacities_file);
    // TODO: frequency bins? // loaded from opacities!

    // stellar spectrum ?
    return true;
}

bool two_streams_radiative_transfer::initialise_memory(
    const ESP&               esp,
    device_RK_array_manager& phy_modules_core_arrays) {

    nlayer = esp.nv; // (??) TODO: check

    // TODO: understand what needs to be stored per column. and what can be global for internal conputation
    // what needs to be passed outside or stored should be global, others can be per column

    float mu_star = 0.0;

    // as set in host_functions.set_up_numerical_parameters
    // w_0_limit
    w_0_limit = 1.0 - 1e-10; 

    // TODO load star flux.
    real_star = false;
    
    alf.set_parameters(nlayer,           // const int&    nlayer_,
		       iso,              // const bool&   iso_,
		       T_star,           // const double& T_star_,
		       real_star,        // const bool&   real_star_,
		       fake_opac,        // const double& fake_opac_,
		       T_surf,           // const double& T_surf_,
		       albedo,           // const double& surf_albedo_,
		       g_0,              // const double& g_0_,
		       epsi,             // const double& epsi_,
		       mu_star,          // const double& mu_star_,
		       scat,             // const bool&   scat_,
		       scat_corr,         // const bool&   scat_corr_,
		       R_planet,         // const double& R_planet_,
		       R_star,           // const double& R_star_,
		       a,                // const double& a_,
		       dir_beam,         // const bool&   dir_beam_,
		       geom_zenith_corr, // const bool&   geom_zenith_corr_,
		       f_factor,         // const double& f_factor_,
		       w_0_limit,        // const double& w_0_limit_,
		       albedo,           // const double& albedo_,
		       i2s_transition,    // const double& i2s_transition_,
		       false);           // const bool&   debug_

    // initialise opacities table -> gives frequency bins
    alf.load_opacities(opacities_file);
    printf("Loaded opacities, using %d bins with %d weights per bin\n", alf.opacities.nbin, alf.opacities.ny);
    // initialise planck tables
    alf.prepare_planck_table();
    printf("Built Planck Table for %d bins, Star temp %g K\n", alf.opacities.nbin, alf.T_star);
    // initialise alf

    // TODO: where to do this, check
    // TODO where does starflux come from?
    // correct_incident_energy 
    // alf.correct_incident_energy(dev_starflux,
    // 				real_star, 
    // 				true);
    
    alf.allocate_internal_variables();

    int ninterface = nlayer + 1;
    int nbin = alf.opacities.nbin;
    int ny = alf.opacities.ny;
    int nlayer_nbin = nlayer * nbin;
    int ninterface_nbin = ninterface*nbin;
    int ninterface_wg_nbin = ninterface*ny*nbin;

    // allocate interface state variables to be interpolated

    pressure_int.allocate(ninterface);
    temperature_int.allocate(ninterface);
    
    // TODO: allocate here. Should be read in in case of real_star == true
    star_flux.allocate(nbin);
    
    F_down_wg.allocate(ninterface_wg_nbin);
    F_up_wg.allocate(ninterface_wg_nbin);
    Fc_down_wg.allocate(ninterface_wg_nbin);
    Fc_up_wg.allocate(ninterface_wg_nbin);
    F_dir_wg.allocate(ninterface_wg_nbin);
    Fc_dir_wg.allocate(ninterface_wg_nbin);
    F_down_tot.allocate(ninterface);
    F_up_tot.allocate(ninterface);
    F_down_band.allocate(ninterface_nbin);
    F_up_band.allocate(ninterface_nbin);
    F_dir_band.allocate(ninterface_nbin);
    F_net.allocate(ninterface);
    F_net_diff.allocate(nlayer);

    g_0_tot_lay.allocate(nlayer_nbin);
    g_0_tot_int.allocate(ninterface_nbin);
    cloud_opac_lay.allocate(nlayer);
    cloud_opac_int.allocate(ninterface);
    cloud_scat_cross_lay.allocate(nlayer_nbin);
    cloud_scat_cross_int.allocate(ninterface_nbin);

    // TODO: currently, realstar = false, no spectrum
    star_flux.zero();

    // TODO: currently, all clouds set to zero. Not used.
    
    g_0_tot_lay.zero();
    g_0_tot_int.zero();
    cloud_opac_lay.zero();
    cloud_opac_int.zero();
    cloud_scat_cross_lay.zero();
    cloud_scat_cross_int.zero();

    bool clouds = false;
    
    alf.set_clouds_data(clouds,
			*cloud_opac_lay,
			*cloud_opac_int,
			*cloud_scat_cross_lay,
			*cloud_scat_cross_int,
			*g_0_tot_lay,
			*g_0_tot_int);
    
    return true;
}

bool two_streams_radiative_transfer::initial_conditions(const ESP&             esp,
                                                        const SimulationSetup& sim,
                                                        storage*               s) {
  // what should be initialised here and what is to initialise at each loop ?
  // what to initialise here and what to do in initialise memory ?
  
  // initialise Alfrodull

  
  
  return true;
}

// initialise delta_colmass arrays from pressure
// same as helios.source.host_functions.construct_grid
__global__ void initialise_delta_colmass(double * delta_col_mass,
					 double * delta_col_mass_upper,
					 double * delta_col_mass_lower,
					 double * pressure_lay,
					 double * pressure_int,
					 double gravit,
					 int num_layers)
{
  int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (layer_idx < num_layers) {
    delta_col_mass[layer_idx]       = (pressure_int[layer_idx] - pressure_int[layer_idx + 1])/gravit;
    delta_col_mass_upper[layer_idx] = (pressure_lay[layer_idx] - pressure_int[layer_idx + 1])/gravit;
    delta_col_mass_lower[layer_idx] = (pressure_int[layer_idx] - pressure_lay[layer_idx])/gravit;
  }
}
					 

// single column pressure and temperature interpolation from layers to interfaces
// needs to loop from 0 to number of interfaces (nvi = nv+1)
// same as profX_RT
__global__ void interpolate_temperature_and_pressure(double * temperature_lay,
						     double * temperature_int,
						     double * pressure_lay,
						     double * pressure_int,
						     double * density,
						     double * altitude_lay,
						     double * altitude_int,
						     double gravit, 
						     int num_layers)
{
  int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (layer_idx == 0) {
      // extrapolate to lower boundary
      double psm = pressure_lay[1]
	- density[0] * gravit * (2 * altitude_int[0] - altitude_lay[0] - altitude_lay[1]);
      
      double ps = 0.5 * (pressure_lay[0] + psm);
      
      pressure_int[0] = ps;
      temperature_int[0] = temperature_lay[0];
    }
    else if (layer_idx == num_layers) {
      // extrapolate to top boundary
      double pp = pressure_lay[num_layers - 2]
	+ (pressure_lay[num_layers - 1] - pressure_lay[num_layers - 2])
	/ (altitude_lay[num_layers - 1] - altitude_lay[num_layers - 2])
	* (2 * altitude_int[num_layers] - altitude_lay[num_layers - 1] - altitude_lay[num_layers - 2]);
      if (pp < 0.0)
	pp = 0.0; //prevents pressure at the top from becoming negative
      double ptop = 0.5 * (pressure_lay[num_layers - 1] + pp);
      
      pressure_int[num_layers] = ptop;
      temperature_int[num_layers] = temperature_lay[num_layers - 1];
    }
    else if (layer_idx < num_layers) {
      // interpolation between layers
      // linear interpolation
      double xi                   = altitude_int[layer_idx];
      double xi_minus             = altitude_lay[layer_idx - 1];
      double xi_plus              = altitude_lay[layer_idx];
      double a                    = (xi - xi_plus) / (xi_minus - xi_plus);
      double b                    = (xi - xi_minus) / (xi_plus - xi_minus);
      
      pressure_int[layer_idx] =
	pressure_lay[layer_idx - 1] * a + pressure_lay[layer_idx] * b;
      temperature_int[layer_idx] =
	temperature_lay[layer_idx - 1] * a + temperature_lay[layer_idx] * b;
    }  
}

__global__ void increment_column_Qheat(double * F_net,  // net flux, layer
				       double * z_int,
				       double * Qheat,
				       int num_layers)
{
  int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer_idx < num_layers) {
    // delta_flux/delta_z
    Qheat[layer_idx] += -(F_net[layer_idx + 1] - F_net[layer_idx])/(z_int[layer_idx + 1] - z_int[layer_idx]);
  }
}

void cuda_check_status_or_exit()
{
  cudaError_t err = cudaGetLastError();

  // Check device query
  if (err != cudaSuccess) {
    log::printf("[%s:%d] CUDA error check reports error: %s\n",
		__FILE__,
		__LINE__,
		cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

bool two_streams_radiative_transfer::phy_loop(ESP&                   esp,
                                              const SimulationSetup& sim,
                                              int                    nstep, // Step number
                                              double                 time_step)             // Time-step [s]
{
  // loop on columns
  for (int column_idx = 0; column_idx < esp.point_num; column_idx++)
    {
      printf("two_stream_rt::phy_loop, step: %d, column: %d\n", nstep, column_idx);
      int num_layers =  esp.nv;

      
      // TODO: get column offset
      int column_offset = column_idx*num_layers;
      

      double gravit = sim.Gravit;
      // fetch column values
      
      // TODO: check that I got the correct ones between slow and fast modes
      double * column_layer_temperature = &(esp.temperature_d[column_offset]);
      double * column_layer_pressure = &(esp.pressure_d[column_offset]);
      double * column_density = &(esp.Rho_d[column_offset]);
      // initialise interpolated T and P

      printf("interpolate_temperature\n");
      const int num_blocks = 256;
      interpolate_temperature_and_pressure<<<(esp.point_num / num_blocks) +1,
	num_blocks>>>(column_layer_temperature,
		      *temperature_int,
		      column_layer_pressure,
		      *pressure_int,
		      column_density,
		      esp.Altitude_d,
		      esp.Altitudeh_d,
		      gravit,
		      num_layers);
      cudaDeviceSynchronize();
      cuda_check_status_or_exit();

      // initialise delta_col_mass
      // TODO: should this go inside alf?
      printf("initialise_delta_colmass\n");
      initialise_delta_colmass<<<(esp.point_num / num_blocks) +1,
	num_blocks>>>(*alf.delta_col_mass,
		      *alf.delta_col_upper,
		      *alf.delta_col_lower,
		      column_layer_pressure,
		      *pressure_int,
		      gravit,
		      num_layers);
      cudaDeviceSynchronize();
      cuda_check_status_or_exit();
	
	// get z_lay
	// TODO: z_lay for beam computation
	// TODO: check how it is used and check that it doesn't interpolate to interface
	//        in which case we need to pass z_int
	double * z_lay = esp.Altitude_d;
	double * z_int = esp.Altitudeh_d;
      // compute mu_star per column
	// TODO: compute mu_star for each column
      
      // internal to alfrodull_engine
      
      // TODO: define star_flux     
      double * dev_starflux = nullptr;
      // limit where to switch from noniso to iso equations to keep model stable
      // as defined in host_functions.set_up_numerical_parameters
      double delta_tau_limit = 1e-4;
      // TODO: add code to skip internal interpolation
	// compute fluxes

      // Check in here, some values from initial setup might change per column: e.g. mu_star;
      printf("compute_radiative_transfer\n");
	    
      alf.compute_radiative_transfer(dev_starflux,               // dev_starflux
				     column_layer_temperature,   // dev_T_lay
				     *temperature_int,           // dev_T_int
				     column_layer_pressure,      // dev_p_lay
				     *pressure_int,              // dev_p_int
				     false,                  // interp_press_and_temp
				     true,                       // interp_and_calc_flux_step
				     z_lay,                      // z_lay
				     false,                      // singlewalk
				     *F_down_wg,
				     *F_up_wg,
				     *Fc_down_wg,
				     *Fc_up_wg,
				     *F_dir_wg,
				     *Fc_dir_wg,
				     delta_tau_limit,
				     *F_down_tot,
				     *F_up_tot,
				     *F_net,
				     *F_down_band,
				     *F_up_band,
				     *F_dir_band);
      cuda_check_status_or_exit();
      



    // compute Delta flux

    // set Qheat
      printf("increment_column_Qheat\n");
      // increment_column_Qheat<<<(esp.point_num / num_blocks) +1,
      // 	num_blocks>>>(*F_net,  // net flux, layer
      // 		      z_int,
      // 		      esp.profx_Qheat_d,
      // 		      num_layers);
      // cudaDeviceSynchronize();
      cuda_check_status_or_exit();
    }
  
    return true;
}

bool two_streams_radiative_transfer::store(const ESP& esp, storage& s) {

  
    return true;
}


bool two_streams_radiative_transfer::free_memory() {

    return true;
}
