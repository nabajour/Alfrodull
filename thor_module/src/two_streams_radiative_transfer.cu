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

#include "alfrodull_engine.h"

#include "insolation_angle.h"

#include "physics_constants.h"


const char PBSTR[] = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
const int  PBWIDTH = 60;

void print_progress(double percentage) {
    int val  = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

two_streams_radiative_transfer::two_streams_radiative_transfer() {
}

two_streams_radiative_transfer::~two_streams_radiative_transfer() {
}

void two_streams_radiative_transfer::print_config() {
    printf("Tstar: %f\n", T_star);
    printf("Alf_iso: %s\n", iso ? "true" : "false");
    printf("Alf_real_star: %s\n", real_star ? "true" : "false");
    printf("Alf_fake_opac: %f\n", fake_opac);

    printf("Alf_T_surf: %f\n", T_surf);
    printf("Alf_albedo: %f\n", albedo);
    printf("Alf_g_0: %f\n", g_0);
    printf("Alf_diffusivity: %f\n", epsi);

    printf("Alf_scat: %s\n", scat ? "true" : "false");
    printf("Alf_scat_corr: %s\n", scat_corr ? "true" : "false");
    printf("R_star: %f [R_SUN]\n", R_star_config);
    printf("planet star dist: %f [au]\n", planet_star_dist_config);

    printf("Alf_dir_beam: %s\n", dir_beam ? "true" : "false");
    printf("Alf_geom_zenith_corr: %s\n", geom_zenith_corr ? "true" : "false");

    printf("Alf_w_0_limit: %f\n", w_0_limit);
    printf("Alf_i2s_transition: %f\n", i2s_transition);
    printf("Alf_opacities_file: %s\n", opacities_file.c_str());
    printf("Alf_compute_every_nstep: %d\n", compute_every_n_iteration);


    // orbit/insolation properties
    printf("    Synchronous rotation        = %s.\n", sync_rot_config ? "true" : "false");
    printf("    Orbital mean motion         = %f rad/s.\n", mean_motion_config);
    printf("    Host star initial right asc = %f deg.\n", alpha_i_config);
    printf("    Planet true initial long    = %f.\n", true_long_i_config);
    printf("    Orbital eccentricity        = %f.\n", ecc_config);
    printf("    Obliquity                   = %f deg.\n", obliquity_config);
    printf("    Longitude of periastron     = %f deg.\n", longp_config);
}

bool two_streams_radiative_transfer::configure(config_file& config_reader) {
    config_reader.append_config_var("Tstar", T_star, T_star);
    config_reader.append_config_var(
        "planet_star_dist", planet_star_dist_config, planet_star_dist_config);
    config_reader.append_config_var("radius_star", R_star_config, R_star_config);

    config_reader.append_config_var("Alf_iso", iso, iso);
    config_reader.append_config_var("Alf_real_star", real_star, real_star);
    config_reader.append_config_var("Alf_fake_opac", fake_opac, fake_opac);
    config_reader.append_config_var("Alf_T_surf", T_surf, T_surf); // ?
    config_reader.append_config_var("Alf_albedo", albedo, albedo);
    config_reader.append_config_var("Alf_g_0", g_0, g_0);
    config_reader.append_config_var("Alf_diffusivity", epsi, epsi);

    config_reader.append_config_var("Alf_scat", scat, scat);
    config_reader.append_config_var("Alf_scat_corr", scat_corr, scat_corr);

    config_reader.append_config_var("Alf_dir_beam", dir_beam, dir_beam);
    config_reader.append_config_var("Alf_geom_zenith_corr", geom_zenith_corr, geom_zenith_corr);
    config_reader.append_config_var("Alf_i2s_transition", i2s_transition, i2s_transition);

    config_reader.append_config_var("Alf_opacities_file", opacities_file, opacities_file);
    config_reader.append_config_var(
        "Alf_compute_every_nstep", compute_every_n_iteration, compute_every_n_iteration);

    // orbit/insolation properties
    config_reader.append_config_var("sync_rot", sync_rot_config, sync_rot_config);
    config_reader.append_config_var("mean_motion", mean_motion_config, mean_motion_config);
    config_reader.append_config_var("alpha_i", alpha_i_config, alpha_i_config);
    config_reader.append_config_var("true_long_i", true_long_i_config, true_long_i_config);
    config_reader.append_config_var("ecc", ecc_config, ecc_config);
    config_reader.append_config_var("obliquity", obliquity_config, obliquity_config);
    config_reader.append_config_var("longp", longp_config, longp_config);

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

    R_star_SI = R_star_config * R_SUN;

    planet_star_dist_SI = planet_star_dist_config * AU;


    // as set in host_functions.set_up_numerical_parameters
    // w_0_limit
    w_0_limit = 1.0 - 1e-10;

    // TODO load star flux.
    real_star = false;

    double f_factor = 1.0;

    alf.set_parameters(nlayer,              // const int&    nlayer_,
                       iso,                 // const bool&   iso_,
                       T_star,              // const double& T_star_,
                       real_star,           // const bool&   real_star_,
                       fake_opac,           // const double& fake_opac_,
                       T_surf,              // const double& T_surf_,
                       albedo,              // const double& surf_albedo_,
                       g_0,                 // const double& g_0_,
                       epsi,                // const double& epsi_,
                       mu_star,             // const double& mu_star_,
                       scat,                // const bool&   scat_,
                       scat_corr,           // const bool&   scat_corr_,
                       0.0,                 // const double& R_planet_, filled in later
                       R_star_SI,           // const double& R_star_,
                       planet_star_dist_SI, // const double& a_,
                       dir_beam,            // const bool&   dir_beam_,
                       geom_zenith_corr,    // const bool&   geom_zenith_corr_,
                       f_factor,            // const double& f_factor_,
                       w_0_limit,           // const double& w_0_limit_,
                       albedo,              // const double& albedo_,
                       i2s_transition,      // const double& i2s_transition_,
                       false);              // const bool&   debug_

    // initialise opacities table -> gives frequency bins
    alf.load_opacities(opacities_file);
    cudaDeviceSynchronize();
    printf("Loaded opacities, using %d bins with %d weights per bin\n",
           alf.opacities.nbin,
           alf.opacities.ny);


    alf.allocate_internal_variables();

    int ninterface         = nlayer + 1;
    int nbin               = alf.opacities.nbin;
    int ny                 = alf.opacities.ny;
    int nlayer_nbin        = nlayer * nbin;
    int ninterface_nbin    = ninterface * nbin;
    int ninterface_wg_nbin = ninterface * ny * nbin;

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
    F_down_tot.allocate(esp.point_num * ninterface);
    F_up_tot.allocate(esp.point_num * ninterface);
    F_down_band.allocate(ninterface_nbin);
    F_up_band.allocate(ninterface_nbin);
    F_dir_band.allocate(esp.point_num*ninterface_nbin);
    // TODO: check, ninterface or nlayers ?
    F_net.allocate(esp.point_num * ninterface);

    g_0_tot_lay.allocate(nlayer_nbin);
    g_0_tot_int.allocate(ninterface_nbin);
    cloud_opac_lay.allocate(nlayer);
    cloud_opac_int.allocate(ninterface);
    cloud_scat_cross_lay.allocate(nlayer_nbin);
    cloud_scat_cross_int.allocate(ninterface_nbin);

    col_mu_star.allocate(esp.point_num);
    col_mu_star.zero();

    Qheat.allocate(esp.point_num * nlayer);

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

    // this is only known here, comes from sim setup.
    alf.R_planet = sim.A;

        // initialise planck tables
    alf.prepare_planck_table();
    printf("Built Planck Table for %d bins, Star temp %g K\n", alf.opacities.nbin, alf.T_star);
    // initialise alf

    // TODO: where to do this, check
    // TODO where does starflux come from?
    // correct_incident_energy
    alf.correct_incident_energy(*star_flux,
     				real_star,
     				true);
    // initialise Alfrodull
    sync_rot = sync_rot_config;
    if (sync_rot) {
        mean_motion = sim.Omega; //just set for the sync_rot, obl != 0 case
    }
    else {
        mean_motion = mean_motion_config;
    }

    true_long_i           = true_long_i_config * M_PI / 180.0;
    longp                 = longp_config * M_PI / 180.0;
    ecc                   = ecc_config;
    double true_anomaly_i = fmod(true_long_i - longp, (2 * M_PI));
    double ecc_anomaly_i  = true2ecc_anomaly(true_anomaly_i, ecc);
    mean_anomaly_i        = fmod(ecc_anomaly_i - ecc * sin(ecc_anomaly_i), (2 * M_PI));
    alpha_i               = alpha_i_config * M_PI / 180.0;
    obliquity             = obliquity_config * M_PI / 180.0;


    return true;
}

// initialise delta_colmass arrays from pressure
// same as helios.source.host_functions.construct_grid
__global__ void initialise_delta_colmass(double* delta_col_mass,
                                         double* delta_col_mass_upper,
                                         double* delta_col_mass_lower,
                                         double* pressure_lay,
                                         double* pressure_int,
                                         double  gravit,
                                         int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer_idx < num_layers) {
        delta_col_mass[layer_idx] =
            (pressure_int[layer_idx] - pressure_int[layer_idx + 1]) / gravit;
        delta_col_mass_upper[layer_idx] =
            (pressure_lay[layer_idx] - pressure_int[layer_idx + 1]) / gravit;
        delta_col_mass_lower[layer_idx] =
            (pressure_int[layer_idx] - pressure_lay[layer_idx]) / gravit;
    }
}


// single column pressure and temperature interpolation from layers to interfaces
// needs to loop from 0 to number of interfaces (nvi = nv+1)
// same as profX_RT
__global__ void interpolate_temperature_and_pressure(double* temperature_lay,
                                                     double* temperature_int,
                                                     double* pressure_lay,
                                                     double* pressure_int,
                                                     double* density,
                                                     double* altitude_lay,
                                                     double* altitude_int,
                                                     double  gravit,
                                                     int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer_idx == 0) {
        // extrapolate to lower boundary
        double psm =
            pressure_lay[1]
            - density[0] * gravit * (2 * altitude_int[0] - altitude_lay[0] - altitude_lay[1]);

        double ps = 0.5 * (pressure_lay[0] + psm);

        pressure_int[0]    = ps;
        temperature_int[0] = temperature_lay[0];
    }
    else if (layer_idx == num_layers) {
        // extrapolate to top boundary
        double pp = pressure_lay[num_layers - 2]
                    + (pressure_lay[num_layers - 1] - pressure_lay[num_layers - 2])
                          / (altitude_lay[num_layers - 1] - altitude_lay[num_layers - 2])
                          * (2 * altitude_int[num_layers] - altitude_lay[num_layers - 1]
                             - altitude_lay[num_layers - 2]);
        if (pp < 0.0)
            pp = 0.0; //prevents pressure at the top from becoming negative
        double ptop = 0.5 * (pressure_lay[num_layers - 1] + pp);

        pressure_int[num_layers]    = ptop;
        temperature_int[num_layers] = temperature_lay[num_layers - 1];
    }
    else if (layer_idx < num_layers) {
        // interpolation between layers
        // linear interpolation
        double xi       = altitude_int[layer_idx];
        double xi_minus = altitude_lay[layer_idx - 1];
        double xi_plus  = altitude_lay[layer_idx];
        double a        = (xi - xi_plus) / (xi_minus - xi_plus);
        double b        = (xi - xi_minus) / (xi_plus - xi_minus);

        pressure_int[layer_idx] = pressure_lay[layer_idx - 1] * a + pressure_lay[layer_idx] * b;
        temperature_int[layer_idx] =
            temperature_lay[layer_idx - 1] * a + temperature_lay[layer_idx] * b;
    }
}

__global__ void increment_Qheat(double* Qheat_global, double* Qheat, int num_sample) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sample) {
        // delta_flux/delta_z
        Qheat_global[idx] += Qheat[idx];
    }
}

__global__ void compute_column_Qheat(double* F_net, // net flux, layer
                                     double* z_int,
                                     double* Qheat,
                                     int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (layer_idx < num_layers) {
        // delta_flux/delta_z
        Qheat[layer_idx] =
            -(F_net[layer_idx + 1] - F_net[layer_idx]) / (z_int[layer_idx + 1] - z_int[layer_idx]);
    }
}

__global__ void compute_col_mu_star(double* col_mu_star,
                                    double* lonlat_d,
                                    double  alpha,
                                    double  alpha_i,
                                    double  sin_decl,
                                    double  cos_decl,
                                    double  ecc,
                                    double  obliquity,
                                    bool    sync_rot,
                                    int     num_points) {
    int column_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (column_idx < num_points) {
        double coszrs = calc_zenith(lonlat_d, //latitude/longitude grid
                                    alpha,    //current RA of star (relative to zero long on planet)
                                    alpha_i,
                                    sin_decl, //declination of star
                                    cos_decl,
                                    sync_rot,
                                    ecc,
                                    obliquity,
                                    column_idx);

	if (coszrs < 0.0)
	  col_mu_star[column_idx] = 0.0;
	else
	  col_mu_star[column_idx] = coszrs;
    }
}

void cuda_check_status_or_exit() {
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
  int nbin = alf.opacities.nbin;
    //  update global insolation properties if necessary
    if (sync_rot) {
        if (ecc > 1e-10) {
            update_spin_orbit(nstep * time_step, sim.Omega);
        }
    }
    else {
        update_spin_orbit(nstep * time_step, sim.Omega);
    }

    const int num_blocks = 256;

    if (nstep % compute_every_n_iteration == 0 || start_up) {
        compute_col_mu_star<<<(esp.point_num / num_blocks) + 1, num_blocks>>>(*col_mu_star,
                                                                              esp.lonlat_d,
                                                                              alpha,
                                                                              alpha_i,
                                                                              sin_decl,
                                                                              cos_decl,
                                                                              ecc,
                                                                              obliquity,
                                                                              sync_rot,
                                                                              esp.point_num);
        cuda_check_status_or_exit();

        std::unique_ptr<double[]> loc_col_mu_star = std::make_unique<double[]>(esp.point_num);
        col_mu_star.fetch(loc_col_mu_star);

        printf("\r\n");
        // loop on columns
        for (int column_idx = 0; column_idx < esp.point_num; column_idx++) {
            print_progress((column_idx + 1.0) / double(esp.point_num));
            //printf("two_stream_rt::phy_loop, step: %d, column: %d\n", nstep, column_idx);
            int num_layers = esp.nv;


            // TODO: get column offset
            int column_offset = column_idx * num_layers;


            double gravit = sim.Gravit;
            // fetch column values

            // TODO: check that I got the correct ones between slow and fast modes
            double* column_layer_temperature = &(esp.temperature_d[column_offset]);
            double* column_layer_pressure    = &(esp.pressure_d[column_offset]);
            double* column_density           = &(esp.Rho_d[column_offset]);
            // initialise interpolated T and P

            //printf("interpolate_temperature\n");

            interpolate_temperature_and_pressure<<<(esp.point_num / num_blocks) + 1, num_blocks>>>(
                column_layer_temperature,
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
            //printf("initialise_delta_colmass\n");
            initialise_delta_colmass<<<(esp.point_num / num_blocks) + 1, num_blocks>>>(
                *alf.delta_col_mass,
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
            double* z_lay = esp.Altitude_d;
            double* z_int = esp.Altitudeh_d;
            // compute mu_star per column
            // TODO: compute mu_star for each column
            double mu_star = loc_col_mu_star[column_idx];
            // internal to alfrodull_engine

            // TODO: define star_flux
            double* dev_starflux = nullptr;
            // limit where to switch from noniso to iso equations to keep model stable
            // as defined in host_functions.set_up_numerical_parameters
            double delta_tau_limit = 1e-4;
            // TODO: add code to skip internal interpolation
            // compute fluxes

            // Check in here, some values from initial setup might change per column: e.g. mu_star;
            //printf("compute_radiative_transfer\n");

            int     ninterface        = nlayer + 1;
            int     column_offset_int = column_idx * ninterface;
            double* F_col_down_tot    = &((*F_down_tot)[column_offset_int]);
            double* F_col_up_tot      = &((*F_up_tot)[column_offset_int]);
            double* F_col_net         = &((*F_net)[column_offset_int]);
	    double* F_dir_band_col    = &((*F_dir_band)[ column_idx * ninterface * nbin]);
            alf.compute_radiative_transfer(dev_starflux,             // dev_starflux
                                           column_layer_temperature, // dev_T_lay
                                           *temperature_int,         // dev_T_int
                                           column_layer_pressure,    // dev_p_lay
                                           *pressure_int,            // dev_p_int
                                           false,                    // interp_press_and_temp
                                           true,                     // interp_and_calc_flux_step
                                           z_lay,                    // z_lay
                                           false,                    // singlewalk
                                           *F_down_wg,
                                           *F_up_wg,
                                           *Fc_down_wg,
                                           *Fc_up_wg,
                                           *F_dir_wg,
                                           *Fc_dir_wg,
                                           delta_tau_limit,
                                           F_col_down_tot,
                                           F_col_up_tot,
                                           F_col_net,
                                           *F_down_band,
                                           *F_up_band,
                                           F_dir_band_col,
                                           mu_star);
            cuda_check_status_or_exit();


            // compute Delta flux

            // set Qheat
            //printf("increment_column_Qheat\n");
            double* qheat = &((*Qheat)[column_offset]);
            compute_column_Qheat<<<(esp.point_num / num_blocks) + 1,
                                   num_blocks>>>(*F_net, // net flux, layer
                                                 z_int,
                                                 qheat,
                                                 num_layers);
            cudaDeviceSynchronize();
            cuda_check_status_or_exit();
        }
        start_up = false;

	
	// std::shared_ptr<double[]> F_dir_band_h = F_dir_band.get_host_data();
	// int lev = (esp.nvi - 1);
	// for (int i = 0; i < esp.point_num; i++) {
	//   printf("i: %d, mu_star: %g\n", i, loc_col_mu_star[i]);
	//   for (int b = 0; b < nbin; b++)
	//     	  printf("\tb: %d, F_dir: %g\n", b, F_dir_band_h[i*esp.nvi*nbin + lev*nbin + b ]);
	// }
	
    }


    int num_samples = (esp.point_num * nlayer);
    increment_Qheat<<<(num_samples / num_blocks) + 1, num_blocks>>>(
         esp.profx_Qheat_d, *Qheat, num_samples);
    cudaDeviceSynchronize();
    cuda_check_status_or_exit();

    // if (nstep * time_step < (2 * M_PI / mean_motion)) {
    //     // stationary orbit/obliquity
    //     // calculate annually average of insolation for the first orbit
    //     annual_insol<<<NBRT, NTH>>>(insol_ann_d, insol_d, nstep, esp.point_num);
    // }


    printf("\r\n");
    return true;
}

bool two_streams_radiative_transfer::store_init(storage& s) {
    s.append_value(T_star, "/Tstar", "K", "Temperature of host star");
    // s.append_value(Tint, "/Tint", "K", "Temperature of interior heat flux");
    s.append_value(
        planet_star_dist_config, "/planet_star_dist", "au", "distance b/w host star and planet");
    s.append_value(R_star_config, "/radius_star", "R_sun", "radius of host star");

    s.append_value(iso ? 1.0 : 0.0, "/alf_isothermal", "-", "Isothermal layers");
    s.append_value(
        real_star ? 1.0 : 0.0, "/alf_real_star", "-", "Alfrodull use real star spectrum or Planck");
    s.append_value(
        fake_opac ? 1.0 : 0.0, "/alf_fake_opac", "-", "Alfrodull use artificial opacity");
    s.append_value(scat ? 1.0 : 0.0, "/alf_scat", "-", "Scattering");
    s.append_value(
        scat_corr ? 1.0 : 0.0, "/alf_scat_corr", "-", "Improved two-stream scattering correction");

    s.append_value(g_0, "/alf_g_0", "-", "asymmetry factor");
    s.append_value(epsi, "/alf_epsi", "-", "Diffusivity factor");


    s.append_value(albedo, "/alf_surface_albedo", "-", "Alfrodull surface albedo");
    s.append_value(i2s_transition, "/alf_i2s_transition", "-", "i2s transition");

    s.append_value(compute_every_n_iteration,
                   "/alf_compute_periodicity",
                   "n",
                   "Alfrodull compute periodicity");
    //s.append_value(opacities_file, "/alf_opacity_file", "path", "Alfrodull opacitiy file used");


    s.append_value(dir_beam ? 1.0 : 0.0, "/alf_dir_beam", "-", "Direct irradiation beam");
    s.append_value(geom_zenith_corr ? 1.0 : 0.0,
                   "/alf_geom_zenith_corr",
                   "-",
                   "Geometric zenith angle correction");

    s.append_value(sync_rot ? 1.0 : 0.0, "/sync_rot", "-", "enforce synchronous rotation");
    s.append_value(mean_motion, "/mean_motion", "rad/s", "orbital mean motion");
    s.append_value(alpha_i * 180 / M_PI, "/alpha_i", "deg", "initial RA of host star");
    s.append_value(
        true_long_i * 180 / M_PI, "/true_long_i", "deg", "initial orbital position of planet");
    s.append_value(ecc, "/ecc", "-", "orbital eccentricity");
    s.append_value(obliquity * 180 / M_PI, "/obliquity", "deg", "tilt of spin axis");
    s.append_value(longp * 180 / M_PI, "/longp", "deg", "longitude of periastron");

    return true;
}

bool two_streams_radiative_transfer::store(const ESP& esp, storage& s) {
    std::shared_ptr<double[]> F_net_h = F_net.get_host_data();
    s.append_table(F_net_h.get(), F_net.get_size(), "/F_net", "W m^-2", "Net Flux");

    std::shared_ptr<double[]> Qheat_h = Qheat.get_host_data();
    s.append_table(Qheat_h.get(), Qheat.get_size(), "/Alf_Qheat", "W m^-3", "Alfrodull Qheat");

    std::shared_ptr<double[]> col_mu_star_h = col_mu_star.get_host_data();
    s.append_table(
        col_mu_star_h.get(), col_mu_star.get_size(), "/col_mu_star", "-", "mu star pre column");

    std::shared_ptr<double[]> F_up_tot_h = F_up_tot.get_host_data();
    s.append_table(
        F_up_tot_h.get(), F_up_tot.get_size(), "/F_up_tot", "W m^-2", "Total upward flux");

    std::shared_ptr<double[]> F_down_tot_h = F_down_tot.get_host_data();
    s.append_table(
        F_down_tot_h.get(), F_down_tot.get_size(), "/F_down_tot", "W m^-2", "Total downward flux");


    return true;
}


bool two_streams_radiative_transfer::free_memory() {

    return true;
}

void two_streams_radiative_transfer::update_spin_orbit(double time, double Omega) {

    // Update the insolation related parameters for spin and orbit
    double ecc_anomaly, true_long;

    mean_anomaly = fmod((mean_motion * time + mean_anomaly_i), (2 * M_PI));

    ecc_anomaly = fmod(solve_kepler(mean_anomaly, ecc), (2 * M_PI));

    r_orb     = calc_r_orb(ecc_anomaly, ecc);
    true_long = fmod((ecc2true_anomaly(ecc_anomaly, ecc) + longp), (2 * M_PI));

    sin_decl = sin(obliquity) * sin(true_long);
    cos_decl = sqrt(1.0 - sin_decl * sin_decl);
    alpha    = -Omega * time + true_long - true_long_i + alpha_i;
}
