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
    config_reader.append_config_var("Alf_w_0_limit", w_0_limit, w_0_limit);
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

  

bool two_streams_radiative_transfer::phy_loop(ESP&                   esp,
                                              const SimulationSetup& sim,
                                              int                    nstep, // Step number
                                              double                 time_step)             // Time-step [s]
{
  for (int column_idx = 0; column_idx < esp.point_num; column_idx++)
    {
      // loop on columns
      // TODO: get column offset
      int column_offset = column_idx;

      int num_layers =  esp.nv;
      double gravit = sim.Gravit;
      // fetch column values
      
      // TODO: check that I got the correct ones between slow and fast modes
      double * column_layer_temperature = &(esp.temperature_d[column_offset]);
      double * column_layer_pressure = &(esp.pressure_d[column_offset]);
      double * column_density = &(esp.Rho_d[column_offset]);
      // initialise interpolated T and P

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

      // initialise delta_col_mass
      // TODO: should this go inside alf?
      initialise_delta_colmass<<<(esp.point_num / num_blocks) +1,
	num_blocks>>>(*alf.delta_col_mass,
		      *alf.delta_col_upper,
		      *alf.delta_col_lower,
		      column_layer_pressure,
		      *pressure_int,
		      gravit,
		      num_layers);
	
	// get z_lay
	// TODO: z_lay for beam computation
	// TODO: check how it is used and check that it doesn't interpolate to interface
	//        in which case we need to pass z_int
	double * z_lay = esp.Altitude_d;
      // compute mu_star per column
	// TODO: compute mu_star for each column
      
      // internal to alfrodull_engine
      
      // TODO: define star_flux     
      double * dev_starflux = nullptr;
      // TODO: check where to define this and how this is used
      double delta_tau_limit = 1e-4;
      // TODO: add code to skip internal interpolation
	// compute fluxes
	compute_radiative_transfer(dev_starflux,               // dev_starflux
				   column_layer_temperature,   // dev_T_lay
				   *temperature_int,           // dev_T_int
				   column_layer_pressure,      // dev_p_lay
				   *pressure_int,              // dev_p_int
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
				   
    // Check in here, some values from initial setup might change per column: e.g. mu_star;
    // compute_radiative_transfer(
    //     double* dev_starflux, // in: pil
    //     double*
    //                 dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    //     double*     dev_T_int,                 // in: it, pii, ioi, mmmi, kii
    //     double*     dev_p_lay,                 // in: io, mmm, kil
    //     double*     dev_p_int,                 // in: ioi, mmmi, kii
    //     const bool& interp_and_calc_flux_step, // done at each step
    //     double*     z_lay,
    //     bool        single_walk, // (?) - TODO: check
    //     double*     F_down_wg,
    //     double*     F_up_wg,
    //     double*     Fc_down_wg,
    //     double*     Fc_up_wg,
    //     double*     F_dir_wg,
    //     double*     Fc_dir_wg,
    //     double      delta_tau_limit,
    //     double*     F_down_tot,
    //     double*     F_up_tot,
    //     double*     F_net,
    //     double*     F_down_band,
    //     double*     F_up_band,
    //     double*     F_dir_band,
    //     double*     gauss_weight);

    // compute Delta flux

    // set Qheat
    }
  
    return true;
}

bool two_streams_radiative_transfer::store(const ESP& esp, storage& s) {

  
    return true;
}


bool two_streams_radiative_transfer::free_memory() {

    return true;
}
