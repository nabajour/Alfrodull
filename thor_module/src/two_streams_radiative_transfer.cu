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

#include "binary_test.h"
#include "debug.h"
#include "debug_helpers.h"

#include "alfrodull_engine.h"

#include "physics_constants.h"

#include "directories.h"
#include "storage.h"

#include <string>

#include <functional>
#include <map>

#include "insolation.h"

USE_BENCHMARK();


using std::string;


// show progress bar
#define COLUMN_LOOP_PROGRESS_BAR

// Dont show column progress bar in comparison mode
#ifdef BENCH_POINT_COMPARE
#    undef COLUMN_LOOP_PROGRESS_BAR
#endif // BENCH_POINT_COMPARE

// debugging printout
//#define DEBUG_PRINTOUT_ARRAYS
// dump TP profile to run in HELIOS for profile comparison
//#define DUMP_HELIOS_TP
// stride for column TP profile dump
#ifdef DUMP_HELIOS_TP
const int HELIOS_TP_STRIDE = 1;
#endif // DUMP_HELIOS_TP

//***************************************************************************************************
// DEBUGGING TOOL: integrate weighted values in binned bands and then integrate over bands

// first simple integration over weights
__global__ void integrate_val_band(double* val_wg,       // in
                                   double* val_band,     // out
                                   double* gauss_weight, // in
                                   int     nbin,
                                   int     num_val,
                                   int     ny) {

    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;


    if (val_idx < num_val && bin_idx < nbin) {
        // set memory to 0.

        val_band[bin_idx + nbin * val_idx] = 0;


        int bin_offset = bin_idx + nbin * val_idx;

        for (int y = 0; y < ny; y++) {
            double w             = gauss_weight[y];
            int    weight_offset = y + ny * bin_idx + ny * nbin * val_idx;

            val_band[bin_offset] += 0.5 * w * val_wg[weight_offset];
        }
    }
}

// simple integration over bins/bands
__global__ void integrate_val_tot(double* val_tot,     // out
                                  double* val_band,    // in
                                  double* deltalambda, // in
                                  int     nbin,
                                  int     num_val) {


    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (val_idx < num_val) {

        val_tot[val_idx] = 0.0;

        for (int bin = 0; bin < nbin; bin++) {
            int band_idx = val_idx * nbin + bin;
            val_tot[val_idx] += val_band[band_idx] * deltalambda[bin];
        }
    }
}

std::shared_ptr<double[]>
integrate_band(double* val, double* gauss_weight, int num_val, int nbin, int ny) {
    cuda_device_memory<double> val_band;

    val_band.allocate(num_val * nbin);

    {
        int  num_levels_per_block = 256 / nbin + 1;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block, nbin);

        integrate_val_band<<<gridsize, blocksize>>>(val,          // in
                                                    *val_band,    // out
                                                    gauss_weight, // in
                                                    nbin,
                                                    num_val,
                                                    ny);

        cudaDeviceSynchronize();
    }
    return val_band.get_host_data();
}

std::shared_ptr<double[]> integrate_wg_band(double* val,
                                            double* gauss_weight,
                                            double* deltalambda,
                                            int     num_val,
                                            int     nbin,
                                            int     ny) {
    cuda_device_memory<double> val_band;
    cuda_device_memory<double> val_tot;

    val_band.allocate(num_val * nbin);

    val_tot.allocate(num_val);

    {
        int  num_levels_per_block = 256 / nbin + 1;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block, nbin);

        integrate_val_band<<<gridsize, blocksize>>>(val,          // in
                                                    *val_band,    // out
                                                    gauss_weight, // in
                                                    nbin,
                                                    num_val,
                                                    ny);

        cudaDeviceSynchronize();
    }

    {
        int  num_levels_per_block = 256;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block);
        integrate_val_tot<<<gridsize, blocksize>>>(*val_tot,    // out
                                                   *val_band,   // in
                                                   deltalambda, // in
                                                   nbin,
                                                   num_val);

        cudaDeviceSynchronize();
    }

    return val_tot.get_host_data();
}
//***************************************************************************************************

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
    log::printf("Tstar: %g\n", T_star);
    log::printf("T_intern: %g", T_internal);
    log::printf("Alf_iso: %s\n", iso ? "true" : "false");
    log::printf("Alf_real_star: %s\n", real_star ? "true" : "false");
    log::printf("Alf_fake_opac: %f\n", fake_opac);

    log::printf("Alf_stellar_spectrum: %s\n", stellar_spectrum_file.c_str());

    log::printf("Alf_thomas: %s\n", thomas ? "true" : "false");
    log::printf("Alf_scat_single_walk: %s\n", scat_single_walk ? "true" : "false");
    log::printf("Alf_exp_opac_offset: %g\n", experimental_opacities_offset);

    log::printf("Alf_g_0: %f\n", g_0);
    log::printf("Alf_diffusivity: %f\n", diffusivity);

    log::printf("Alf_G_pm_max_limiter: %s\n", G_pm_limiter ? "true" : "false");
    log::printf("Alf_G_pm_denom_limit: %f\n", G_pm_denom_limit);
    log::printf("Alf_G_pm_mu_star_increment: %f\n", mu_star_wiggle_increment);

    log::printf("Alf_scat: %s\n", scat ? "true" : "false");
    log::printf("Alf_scat_corr: %s\n", scat_corr ? "true" : "false");
    log::printf("R_star: %f [R_SUN]\n", R_star_config);
    log::printf("planet star dist: %f [au]\n", planet_star_dist_config);

    log::printf("Alf_dir_beam: %s\n", dir_beam ? "true" : "false");
    log::printf("Alf_geom_zenith_corr: %s\n", geom_zenith_corr ? "true" : "false");

    log::printf("Alf_w_0_limit: %f\n", w_0_limit);
    log::printf("Alf_i2s_transition: %f\n", i2s_transition);
    log::printf("Alf_opacities_file: %s\n", opacities_file.c_str());
    log::printf("Alf_compute_every_nstep: %d\n", compute_every_n_iteration);

    // spinup-spindown parameters
    log::printf("    Spin up start step          = %d.\n", spinup_start_step);
    log::printf("    Spin up stop step           = %d.\n", spinup_stop_step);
    log::printf("    Spin down start step        = %d.\n", spindown_start_step);
    log::printf("    Spin down stop step         = %d.\n", spindown_stop_step);
}

bool two_streams_radiative_transfer::configure(config_file& config_reader) {
    // variables reused from DG
    config_reader.append_config_var("Tstar", T_star, T_star);
    config_reader.append_config_var("Tint", T_internal, T_internal);
    config_reader.append_config_var(
        "planet_star_dist", planet_star_dist_config, planet_star_dist_config);
    config_reader.append_config_var("radius_star", R_star_config, R_star_config);

    config_reader.append_config_var("Alf_thomas", thomas, thomas);
    config_reader.append_config_var("Alf_scat_single_walk", scat_single_walk, scat_single_walk);
    config_reader.append_config_var(
        "Alf_exp_opac_offset", experimental_opacities_offset, experimental_opacities_offset);
    config_reader.append_config_var("Alf_iso", iso, iso);
    config_reader.append_config_var("Alf_real_star", real_star, real_star);
    config_reader.append_config_var(
        "Alf_stellar_spectrum", stellar_spectrum_file, stellar_spectrum_file);
    config_reader.append_config_var("Alf_fake_opac", fake_opac, fake_opac);

    config_reader.append_config_var("Alf_g_0", g_0, g_0);
    config_reader.append_config_var("Alf_diffusivity", diffusivity, diffusivity);
    config_reader.append_config_var("Alf_G_pm_max_limiter", G_pm_limiter, G_pm_limiter);
    config_reader.append_config_var("Alf_G_pm_denom_limit", G_pm_denom_limit, G_pm_denom_limit);
    config_reader.append_config_var(
        "Alf_G_pm_mu_star_increment", mu_star_wiggle_increment, mu_star_wiggle_increment);

    config_reader.append_config_var("Alf_scat", scat, scat);
    config_reader.append_config_var("Alf_scat_corr", scat_corr, scat_corr);

    config_reader.append_config_var("Alf_dir_beam", dir_beam, dir_beam);
    config_reader.append_config_var("Alf_geom_zenith_corr", geom_zenith_corr, geom_zenith_corr);
    config_reader.append_config_var("Alf_i2s_transition", i2s_transition, i2s_transition);

    config_reader.append_config_var("Alf_opacities_file", opacities_file, opacities_file);
    config_reader.append_config_var(
        "Alf_compute_every_nstep", compute_every_n_iteration, compute_every_n_iteration);

    // spin up spin down
    config_reader.append_config_var("Alf_spinup_start", spinup_start_step, spinup_start_step);
    config_reader.append_config_var("Alf_spinup_stop", spinup_stop_step, spinup_stop_step);
    config_reader.append_config_var("Alf_spindown_start", spindown_start_step, spindown_start_step);
    config_reader.append_config_var("Alf_spindown_stop", spindown_stop_step, spindown_stop_step);

    return true;
}


bool two_streams_radiative_transfer::initialise_memory(
    const ESP&               esp,
    device_RK_array_manager& phy_modules_core_arrays) {
    bool out = true;
    nlayer   = esp.nv; // (??) TODO: check

    // TODO: understand what needs to be stored per column. and what can be global for internal conputation
    // what needs to be passed outside or stored should be global, others can be per column

    float mu_star = 0.0;

    R_star_SI = R_star_config * R_SUN;

    planet_star_dist_SI = planet_star_dist_config * AU;


    // as set in host_functions.set_up_numerical_parameters
    // w_0_limit
    w_0_limit = 1.0 - 1e-14;

    double f_factor = 1.0;

    epsi = 1.0 / diffusivity;

    alf.thomas = thomas;

    alf.G_pm_limiter             = G_pm_limiter;
    alf.G_pm_denom_limit         = G_pm_denom_limit;
    alf.mu_star_wiggle_increment = mu_star_wiggle_increment;

    alf.set_parameters(nlayer,              // const int&    nlayer_,
                       iso,                 // const bool&   iso_,
                       T_star,              // const double& T_star_,
                       real_star,           // const bool&   real_star_,
                       fake_opac,           // const double& fake_opac_,
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
                       i2s_transition,      // const double& i2s_transition_,
                       false);              // const bool&   debug_

    // initialise opacities table -> gives frequency bins
    // set opacity offset for test
    alf.set_experimental_opacity_offset(experimental_opacities_offset);

    alf.load_opacities(opacities_file);
    cudaDeviceSynchronize();
    log::printf("Loaded opacities, using %d bins with %d weights per bin\n",
                alf.opacities.nbin,
                alf.opacities.ny);

    alf.allocate_internal_variables();

    int ninterface         = nlayer + 1;
    int nlayer_plus1       = nlayer + 1;
    int nbin               = alf.opacities.nbin;
    int ny                 = alf.opacities.ny;
    int nlayer_nbin        = nlayer * nbin;
    int ninterface_nbin    = ninterface * nbin;
    int ninterface_wg_nbin = ninterface * ny * nbin;

    if (real_star) {
        // load star flux.
        std::printf("Using Stellar Flux file %s\n", stellar_spectrum_file.c_str());
        star_flux.allocate(nbin);
        if (!path_exists(stellar_spectrum_file)) {
            log::printf("Stellar spectrum file not found: %s\n", stellar_spectrum_file.c_str());
            exit(EXIT_FAILURE);
        }

        double lambda_spectrum_scale = 1e-2;
        double flux_scale            = 1e-1;

        storage s(stellar_spectrum_file, true);
        if (s.has_table("wavelength") && s.has_table("flux")) {
            std::unique_ptr<double[]> lambda_ptr  = nullptr;
            int                       lambda_size = 0;

            std::unique_ptr<double[]> flux_ptr  = nullptr;
            int                       flux_size = 0;

            s.read_table("wavelength", lambda_ptr, lambda_size);
            s.read_table("flux", flux_ptr, flux_size);

            if (lambda_size != nbin || lambda_size != flux_size) {
                log::printf("Wrong size for stellar size arrays\n");
                log::printf("Lambda: %d\n", lambda_size);
                log::printf("Flux: %d\n", flux_size);
                log::printf("nbin: %d\n", nbin);
                exit(EXIT_FAILURE);
            }

            bool                      lambda_check = true;
            double                    epsilon      = 1e-4;
            std::shared_ptr<double[]> star_flux_h  = star_flux.get_host_data_ptr();
            for (int i = 0; i < nbin; i++) {
                star_flux_h[i] = flux_ptr[i] * flux_scale;
                bool check =
                    fabs(lambda_ptr[i] * lambda_spectrum_scale - alf.opacities.data_opac_wave[i])
                        / alf.opacities.data_opac_wave[i]
                    < epsilon;

                if (!check)
                    printf("Missmatch in wavelength at idx [%d] l_spectrum(%g) != l_opac(%g) \n",
                           i,
                           lambda_ptr[i] * lambda_spectrum_scale,
                           alf.opacities.data_opac_wave[i]);
                lambda_check &= check;
            }

            star_flux.put();

            if (!lambda_check) {
                log::printf("wavelength points mismatch between stellar spectrum and opacities\n");
                exit(EXIT_FAILURE);
            }
        }
        else {
            log::printf("table wavelength or flux not found in stellar flux file\n");
            exit(EXIT_FAILURE);
        }
        printf("Stellar flux loaded\n");
    }

    // TODO: allocate here. Should be read in in case of real_star == true
    //    star_flux.allocate(nbin);
    // allocate interface state variables to be interpolated


    pressure_int.allocate(ninterface);
    temperature_int.allocate(ninterface);
    temperature_lay.allocate(nlayer_plus1);


    F_down_wg.allocate(ninterface_wg_nbin);
    F_up_wg.allocate(ninterface_wg_nbin);
    F_dir_wg.allocate(ninterface_wg_nbin);

    if (!iso) {
        Fc_down_wg.allocate(ninterface_wg_nbin);
        Fc_up_wg.allocate(ninterface_wg_nbin);
        Fc_dir_wg.allocate(ninterface_wg_nbin);
    }

    F_down_tot.allocate(esp.point_num * ninterface);
    F_up_tot.allocate(esp.point_num * ninterface);
    F_dir_tot.allocate(esp.point_num * ninterface);
    F_down_band.allocate(ninterface_nbin);
    F_up_band.allocate(ninterface_nbin);
    F_dir_band.allocate(ninterface_nbin);
    // TODO: check, ninterface or nlayers ?
    F_net.allocate(esp.point_num * ninterface);

    F_up_TOA_spectrum.allocate(esp.point_num * nbin);

    g_0_tot_lay.allocate(nlayer_nbin);
    g_0_tot_int.allocate(ninterface_nbin);
    cloud_opac_lay.allocate(nlayer);
    cloud_opac_int.allocate(ninterface);
    cloud_scat_cross_lay.allocate(nlayer_nbin);
    cloud_scat_cross_int.allocate(ninterface_nbin);

    Qheat.allocate(esp.point_num * nlayer);

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

    cudaError_t err = cudaGetLastError();

    // Check device query
    if (err != cudaSuccess) {
        log::printf("[%s:%d] CUDA error check reports error: %s\n",
                    __FILE__,
                    __LINE__,
                    cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef BENCHMARKING
    std::map<string, output_def> debug_arrays = {
        {"F_net", {F_net.ptr_ref(), esp.point_num * ninterface, "Fnet", "Fn", true, dummy}},

        {"F_up_tot",
         {F_up_tot.ptr_ref(), esp.point_num * ninterface, "Fuptot", "Fut", true, dummy}},
        {"F_down_tot",
         {F_down_tot.ptr_ref(), esp.point_num * ninterface, "Fdowntot", "Fdt", true, dummy}},
        {"F_up_wg", {F_up_wg.ptr_ref(), ninterface_wg_nbin, "Fupwg", "Fuw", true, dummy}},
        {"F_down_wg", {F_down_wg.ptr_ref(), ninterface_wg_nbin, "Fdownwg", "Fdw", true, dummy}},
        {"F_up_band", {F_up_band.ptr_ref(), ninterface_nbin, "Fupband", "Fub", true, dummy}},
        {"F_down_band", {F_down_band.ptr_ref(), ninterface_nbin, "Fdownband", "Fdb", true, dummy}},
        {"F_dir_wg", {F_dir_wg.ptr_ref(), ninterface_wg_nbin, "Fdirwg", "Fdirw", true, dummy}},

        {"F_dir_band", {F_dir_band.ptr_ref(), ninterface_nbin, "Fdirband", "Fdib", true, dummy}},


        {"T_lay", {temperature_lay.ptr_ref(), nlayer_plus1, "T_lay", "Tl", true, dummy}},
        {"T_int", {temperature_int.ptr_ref(), ninterface, "T_int", "Ti", true, dummy}},
        {"P_int", {pressure_int.ptr_ref(), ninterface, "P_int", "Pi", true, dummy}},

        //        {"col_mu_star", {col_mu_star.ptr_ref(), esp.point_num, "col_mu_star", "cMu", true, dummy}},
        {"AlfQheat", {Qheat.ptr_ref(), esp.point_num * nlayer, "AlfQheat", "aQh", true, dummy}}};

    BENCH_POINT_REGISTER_PHY_VARS(debug_arrays, (), ());
#endif // BENCHMARKING
    return out;
}

bool two_streams_radiative_transfer::initial_conditions(const ESP&             esp,
                                                        const SimulationSetup& sim,
                                                        storage*               s) {
    if (spinup_start_step > -1 || spinup_stop_step > -1) {
        if (spinup_stop_step < spinup_start_step)
            printf("Alf: inconsistent spinup_start (%d) and spinup_stop (%d) values\n",
                   spinup_start_step,
                   spinup_stop_step);
    }
    if (spindown_start_step > -1 || spindown_stop_step > -1) {
        if (spindown_stop_step < spindown_start_step)
            printf("Alf: inconsistent spindown_start (%d) and spindown_stop (%d) values\n",
                   spindown_start_step,
                   spindown_stop_step);
    }

    bool out = true;
    // what should be initialised here and what is to initialise at each loop ?
    // what to initialise here and what to do in initialise memory ?

    // this is only known here, comes from sim setup.
    alf.R_planet = sim.A;
    cuda_check_status_or_exit(__FILE__, __LINE__);
    // initialise planck tables
    alf.prepare_planck_table();
    log::printf("Built Planck Table for %d bins, Star temp %g K\n", alf.opacities.nbin, alf.T_star);
    // initialise alf

    // TODO: where to do this, check
    // TODO where does starflux come from?
    // correct_incident_energy

    alf.correct_incident_energy(*star_flux, real_star, true);

    // internal flux from internal temperature
    F_intern = STEFANBOLTZMANN * pow(T_internal, 4);

    cuda_check_status_or_exit(__FILE__, __LINE__);

    // request insolation computation
    esp.insolation.set_require();
    return out;
}

// initialise delta_colmass arrays from pressure
// same as helios.source.host_functions.construct_grid
__global__ void initialise_delta_colmass_noniso(double* delta_col_mass_upper,
                                                double* delta_col_mass_lower,
                                                double* pressure_lay,
                                                double* pressure_int,
                                                double  gravit,
                                                int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer_idx < num_layers) {
        delta_col_mass_upper[layer_idx] =
            (pressure_lay[layer_idx] - pressure_int[layer_idx + 1]) / gravit;
        delta_col_mass_lower[layer_idx] =
            (pressure_int[layer_idx] - pressure_lay[layer_idx]) / gravit;
    }
}

// initialise delta_colmass arrays from pressure
// same as helios.source.host_functions.construct_grid
__global__ void initialise_delta_colmass_iso(double* delta_col_mass,
                                             double* pressure_int,
                                             double  gravit,
                                             int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer_idx < num_layers) {
        delta_col_mass[layer_idx] =
            (pressure_int[layer_idx] - pressure_int[layer_idx + 1]) / gravit;
    }
}


// single column pressure and temperature interpolation from layers to interfaces
// needs to loop from 0 to number of interfaces (nvi = nv+1)
// same as profX_RT
__global__ void interpolate_temperature_and_pressure(double* temperature_lay,      // out
                                                     double* temperature_lay_thor, // in
                                                     double* temperature_int,      // out
                                                     double* pressure_lay,         // in
                                                     double* pressure_int,         // out
                                                     double* density,              // in
                                                     double* altitude_lay,         // in
                                                     double* altitude_int,         // in
                                                     double  T_intern,
                                                     double  gravit,
                                                     int     num_layers) {
    int int_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Prepare temperature array with T_intern
    // TODO: check this, HELIOS uses temperature_lay[nlayer] as BOA value (also indexed as temperture_lay[numinterfaces - 1])
    // try helios style
    // printf("-intidx: %d/%d\n", int_idx, num_layers);
    if (int_idx < num_layers) {
        // printf("intidx: %d/%d\n", int_idx, num_layers);
        temperature_lay[int_idx] = temperature_lay_thor[int_idx];
    }
    else if (int_idx == num_layers) {
        //printf("intidx: %d/%d %g *\n", int_idx, num_layers, T_intern);
        temperature_lay[num_layers] = T_intern;
    }

    // compute interface values
    if (int_idx == 0) {
        // extrapolate to lower boundary
        double psm =
            pressure_lay[1]
            - density[0] * gravit * (2 * altitude_int[0] - altitude_lay[0] - altitude_lay[1]);

        double ps = 0.5 * (pressure_lay[0] + psm);

        pressure_int[0]    = ps;
        temperature_int[0] = T_intern;
    }
    else if (int_idx == num_layers) {
        // extrapolate to top boundary
        double pp = pressure_lay[num_layers - 2]
                    + (pressure_lay[num_layers - 1] - pressure_lay[num_layers - 2])
                          / (altitude_lay[num_layers - 1] - altitude_lay[num_layers - 2])
                          * (2 * altitude_int[num_layers] - altitude_lay[num_layers - 1]
                             - altitude_lay[num_layers - 2]);
        if (pp < 0.0)
            pp = 0.0; //prevents pressure at the top from becoming negative
        double ptop = 0.5 * (pressure_lay[num_layers - 1] + pp);

        pressure_int[num_layers] = ptop;
        // extrapolate to top interface
        temperature_int[num_layers] =
            temperature_lay_thor[num_layers - 1]
            + 0.5 * (temperature_lay_thor[num_layers - 1] - temperature_lay_thor[num_layers - 2]);
    }
    else if (int_idx < num_layers) {
        // interpolation between layers
        // Helios computes gy taking the middle between the layers. We can have non uniform Z levels,
        // so linear interpolation
        double xi       = altitude_int[int_idx];
        double xi_minus = altitude_lay[int_idx - 1];
        double xi_plus  = altitude_lay[int_idx];
        double a        = (xi - xi_plus) / (xi_minus - xi_plus);
        double b        = (xi - xi_minus) / (xi_plus - xi_minus);

        pressure_int[int_idx] = pressure_lay[int_idx - 1] * a + pressure_lay[int_idx] * b;

        temperature_int[int_idx] =
            temperature_lay_thor[int_idx - 1] * a + temperature_lay_thor[int_idx] * b;
    }
}

__global__ void
increment_Qheat(double* Qheat_global, double* Qheat, double scaling, int num_sample) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sample) {
        // delta_flux/delta_z
        Qheat_global[idx] += scaling * Qheat[idx];
    }
}

__global__ void compute_column_Qheat(double* F_net, // net flux, layer
                                     double* z_int,
                                     double* Qheat,
                                     double  F_intern,
                                     int     num_layers) {
    int layer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (layer_idx == 0) {
        // delta_flux/delta_z
        // F_net positive in upward direction (F_up - F_down)
        // F_intern positive, flux out of bottom surface
        // Qheat negative when net flux differential out of layer is positive
        Qheat[layer_idx] = -((F_net[1] - F_net[0])) / (z_int[1] - z_int[0]);
    }
    else if (layer_idx < num_layers) {
        // delta_flux/delta_z
        Qheat[layer_idx] =
            -(F_net[layer_idx + 1] - F_net[layer_idx]) / (z_int[layer_idx + 1] - z_int[layer_idx]);
    }
}


bool two_streams_radiative_transfer::phy_loop(ESP&                   esp,
                                              const SimulationSetup& sim,
                                              int                    nstep, // Step number
                                              double                 time_step)             // Time-step [s]
{
    bool run      = true;
    qheat_scaling = 1.0;

    if (spinup_start_step > -1 && spinup_stop_step > -1) {
        if (nstep < spinup_start_step) // before spinup
        {
            run           = false;
            qheat_scaling = 0.0;
        }
        else if ((nstep >= spinup_start_step) && (nstep <= spinup_stop_step)) // during spinup
        {
            double x = (double)(nstep - spinup_start_step)
                       / (double)(spinup_stop_step - spinup_start_step);
            qheat_scaling = (1 + sin(M_PI * x - M_PI / 2.0)) / 2.0;
            run           = true;
        }
    }

    if (spindown_start_step > -1 && spindown_stop_step > -1) {
        if ((nstep >= spindown_start_step) && (nstep <= spindown_stop_step)) {
            double x = (double)(nstep - spindown_start_step)
                       / (double)(spindown_stop_step - spindown_start_step);
            qheat_scaling = 1.0 - (1 + sin(M_PI * x - M_PI / 2.0)) / 2.0;
            run           = true;
        }
        else if (nstep >= spindown_stop_step) {
            run           = false;
            qheat_scaling = 0.0;
        }
    }

    if (run) {

        alf.debug_nstep = nstep;

        const int num_blocks = 256;

        if (nstep % compute_every_n_iteration == 0 || start_up) {
            std::shared_ptr<double[]> col_cos_zenith_angle_h =
                esp.insolation.get_host_cos_zenith_angles();

            Qheat.zero();
            F_down_tot.zero();
            F_up_tot.zero();
            F_dir_tot.zero();
            F_up_band.zero();
            F_dir_band.zero();
            F_net.zero();

            printf("\r\n");
            printf("\r\n");
            printf("\r\n");

            int nbin = alf.opacities.nbin;
            // loop on columns
            for (int column_idx = 0; column_idx < esp.point_num; column_idx++) {
                alf.debug_col_idx = column_idx;

#ifdef COLUMN_LOOP_PROGRESS_BAR
                print_progress((column_idx + 1.0) / double(esp.point_num));
#endif // COLUMN_LOOP_PROGRESS_BAR

                F_up_wg.zero();
                F_down_wg.zero();
                F_dir_wg.zero();
                if (iso) {
                }
                else {
                    Fc_down_wg.zero();
                    Fc_up_wg.zero();

                    Fc_dir_wg.zero();
                }


                alf.reset();

                pressure_int.zero();
                temperature_int.zero();
                temperature_lay.zero();

                g_0_tot_lay.zero();
                g_0_tot_int.zero();
                cloud_opac_lay.zero();
                cloud_opac_int.zero();
                cloud_scat_cross_lay.zero();
                cloud_scat_cross_int.zero();
                int num_layers = esp.nv;


                // TODO: get column offset
                int column_offset = column_idx * num_layers;


                double gravit = sim.Gravit;
                // fetch column values

                // TODO: check that I got the correct ones between slow and fast modes
                double* column_layer_temperature_thor = &(esp.temperature_d[column_offset]);
                double* column_layer_pressure         = &(esp.pressure_d[column_offset]);
                double* column_density                = &(esp.Rho_d[column_offset]);
                // initialise interpolated T and P

                // use mu_star per column
                double mu_star = -col_cos_zenith_angle_h[column_idx];

#ifdef DUMP_HELIOS_TP


                // dump a TP profile for HELIOS input
                if (column_idx % HELIOS_TP_STRIDE == 0) {
                    std::string DBG_OUTPUT_DIR = esp.get_output_dir()
                                                 + "/alfprof/"
                                                   "step_"
                                                 + std::to_string(nstep) + "/column_"
                                                 + std::to_string(column_idx) + "/";
                    create_output_dir(DBG_OUTPUT_DIR);

                    double                    lon = esp.lonlat_h[column_idx * 2 + 0] * 180 / M_PI;
                    double                    lat = esp.lonlat_h[column_idx * 2 + 1] * 180 / M_PI;
                    std::shared_ptr<double[]> pressure_h =
                        get_cuda_data(column_layer_pressure, esp.nv);
                    std::shared_ptr<double[]> temperature_h =
                        get_cuda_data(column_layer_temperature_thor, esp.nv);


                    double p_toa = pressure_h[esp.nv - 1];
                    double p_boa = pressure_h[0];


                    // Print out initial TP profile
                    string output_file_name = DBG_OUTPUT_DIR + "tpprofile_init.dat";

                    FILE*  tp_output_file = fopen(output_file_name.c_str(), "w");
                    string comment = "# Helios TP profile table at lat: [" + std::to_string(lon)
                                     + "] lon: [" + std::to_string(lat) + "] mustar: ["
                                     + std::to_string(mu_star) + "] P_BOA: ["
                                     + std::to_string(p_boa) + "] P_TOA: [" + std::to_string(p_toa)
                                     + "]\n";

                    fprintf(tp_output_file, comment.c_str());
                    fprintf(tp_output_file, "#\tT[K]\tP[bar]\n");

                    for (int i = 0; i < esp.nv; i++) {
                        fprintf(tp_output_file,
                                "%#.6g\t%#.6g\n",
                                temperature_h[i],
                                pressure_h[i] / 1e5);
                    }

                    fclose(tp_output_file);
                }
#endif // DUMP_HELIOS_TP

                interpolate_temperature_and_pressure<<<((num_layers + 1) / num_blocks) + 1,
                                                       num_blocks>>>(*temperature_lay,
                                                                     column_layer_temperature_thor,
                                                                     *temperature_int,
                                                                     column_layer_pressure,
                                                                     *pressure_int,
                                                                     column_density,
                                                                     esp.Altitude_d,
                                                                     esp.Altitudeh_d,
                                                                     T_internal,
                                                                     gravit,
                                                                     num_layers);
                cudaDeviceSynchronize();
                cuda_check_status_or_exit(__FILE__, __LINE__);

                BENCH_POINT_I_S_PHY(
                    nstep, column_idx, "Alf_interpTnP", (), ("T_lay", "T_int", "P_int"));

#ifdef DUMP_HELIOS_TP
                // dump a TP profile for HELIOS input
                if (column_idx % HELIOS_TP_STRIDE == 0) {
                    std::string DBG_OUTPUT_DIR = esp.get_output_dir()
                                                 + "/alfprof/"
                                                   "step_"
                                                 + std::to_string(nstep) + "/column_"
                                                 + std::to_string(column_idx) + "/";
                    create_output_dir(DBG_OUTPUT_DIR);

                    double lon = esp.lonlat_h[column_idx * 2 + 0] * 180 / M_PI;
                    double lat = esp.lonlat_h[column_idx * 2 + 1] * 180 / M_PI;

                    // get col mu star from zenith angle

                    std::shared_ptr<double[]> pressure_int_h    = pressure_int.get_host_data();
                    std::shared_ptr<double[]> temperature_int_h = temperature_int.get_host_data();


                    double p_toa = pressure_int_h[esp.nvi - 1];
                    double p_boa = pressure_int_h[0];


                    // Print out initial TP profile
                    string output_file_name = DBG_OUTPUT_DIR + "tpprofile_interface.dat";

                    FILE*  tp_output_file = fopen(output_file_name.c_str(), "w");
                    string comment        = "# Helios TP interface profile table at lat: ["
                                     + std::to_string(lon) + "] lon: [" + std::to_string(lat)
                                     + "] mustar: [" + std::to_string(mu_star) + "] P_BOA: ["
                                     + std::to_string(p_boa) + "] P_TOA: [" + std::to_string(p_toa)
                                     + "]\n";

                    fprintf(tp_output_file, comment.c_str());
                    fprintf(tp_output_file, "#\tT[K]\tP[bar]\n");


                    for (int i = 0; i < esp.nvi; i++) {
                        fprintf(tp_output_file,
                                "%#.6g\t%#.6g\n",
                                temperature_int_h[i],
                                pressure_int_h[i] / 1e5);
                    }

                    fclose(tp_output_file);
                }
#endif // DUMP_HELIOS_TP

                // initialise delta_col_mass
                // TODO: should this go inside alf?
                // printf("initialise_delta_colmass\n");
                if (iso) {
                    initialise_delta_colmass_iso<<<((num_layers + 1) / num_blocks) + 1,
                                                   num_blocks>>>(
                        *alf.delta_col_mass, *pressure_int, gravit, num_layers);
                }
                else {
                    initialise_delta_colmass_noniso<<<((num_layers + 1) / num_blocks) + 1,
                                                      num_blocks>>>(*alf.delta_col_upper,
                                                                    *alf.delta_col_lower,
                                                                    column_layer_pressure,
                                                                    *pressure_int,
                                                                    gravit,
                                                                    num_layers);
                }
                cudaDeviceSynchronize();
                cuda_check_status_or_exit(__FILE__, __LINE__);
                // printf("initialise_delta_colmass done\n");

                // get z_lay
                // TODO: z_lay for beam computation
                // TODO: check how it is used and check that it doesn't interpolate to interface
                //        in which case we need to pass z_int
                double* z_lay = esp.Altitude_d;
                double* z_int = esp.Altitudeh_d;
                // internal to alfrodull_engine

                double* dev_starflux = *star_flux;
                // limit where to switch from noniso to iso equations to keep model stable
                // as defined in host_functions.set_up_numerical_parameters
                double delta_tau_limit = 1e-4;

                // compute fluxes

                // Check in here, some values from initial setup might change per column: e.g. mu_star;
                //printf("compute_radiative_transfer\n");


                // singlewalk
                //  true -> 201 iterations,
                //  false -> 4 iterations,

                bool    singlewalk_loc    = scat_single_walk;
                int     ninterface        = nlayer + 1;
                int     column_offset_int = column_idx * ninterface;
                double* F_col_down_tot    = &((*F_down_tot)[column_offset_int]);
                double* F_col_up_tot      = &((*F_up_tot)[column_offset_int]);
                double* F_col_dir_tot     = &((*F_dir_tot)[column_offset_int]);
                double* F_col_net         = &((*F_net)[column_offset_int]);
                //            double* F_dir_band_col    = &((*F_dir_band)[ninterface * nbin]);
                double* F_dir_band_col = &((*F_dir_band)[0]);

                double* F_up_TOA_spectrum_col = &((*F_up_TOA_spectrum)[column_idx * nbin]);

                alf.compute_radiative_transfer(dev_starflux,          // dev_starflux
                                               *temperature_lay,      // dev_T_lay
                                               *temperature_int,      // dev_T_int
                                               column_layer_pressure, // dev_p_lay
                                               *pressure_int,         // dev_p_int
                                               false,                 // interp_press_and_temp
                                               true,                  // interp_and_calc_flux_step
                                               z_lay,                 // z_lay
                                               singlewalk_loc,        // singlewalk
                                               *F_down_wg,
                                               *F_up_wg,
                                               *Fc_down_wg,
                                               *Fc_up_wg,
                                               *F_dir_wg,
                                               *Fc_dir_wg,
                                               delta_tau_limit,
                                               F_col_down_tot,
                                               F_col_up_tot,
                                               F_col_dir_tot,
                                               F_col_net,
                                               *F_down_band,
                                               *F_up_band,
                                               F_dir_band_col,
                                               F_up_TOA_spectrum_col,
                                               mu_star);
                cudaDeviceSynchronize();
                cuda_check_status_or_exit(__FILE__, __LINE__);


                // compute Delta flux

                // set Qheat
                //printf("increment_column_Qheat\n");
                double* qheat = &((*Qheat)[column_offset]);
                compute_column_Qheat<<<(esp.nv / num_blocks) + 1,
                                       num_blocks>>>(F_col_net, // net flux, layer
                                                     z_int,
                                                     qheat,
                                                     F_intern,
                                                     num_layers);
                cudaDeviceSynchronize();
                cuda_check_status_or_exit(__FILE__, __LINE__);

#ifdef DEBUG_PRINTOUT_ARRAYS
                debug_print_columns(esp, loc_col_mu_star[column_idx], nstep, column_idx);
#endif // DEBUG_PRINTOUT_ARRAYS
            }
            start_up = false;
        }

        printf("\r\n");

        int num_samples = (esp.point_num * nlayer);
        increment_Qheat<<<(num_samples / num_blocks) + 1, num_blocks>>>(
            esp.profx_Qheat_d, *Qheat, qheat_scaling, num_samples);
        cudaDeviceSynchronize();
        cuda_check_status_or_exit(__FILE__, __LINE__);
    }
    last_step = nstep;

    BENCH_POINT_I_PHY(nstep, "Alf_phy_loop_E", (), ("F_up_tot", "F_down_tot", "AlfQheat"));

    return true;
}

bool two_streams_radiative_transfer::store_init(storage& s) {
    if (!s.has_table("/Tstar"))
        s.append_value(T_star, "/Tstar", "K", "Temperature of host star");
    // s.append_value(Tint, "/Tint", "K", "Temperature of interior heat flux");
    if (!s.has_table("/planet_star_dist"))
        s.append_value(planet_star_dist_config,
                       "/planet_star_dist",
                       "au",
                       "distance b/w host star and planet");

    if (!s.has_table("/radius_star"))
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
    s.append_value(diffusivity, "/alf_diffusivity", "-", "Diffusivity factor");
    s.append_value(epsi, "/alf_epsi", "-", "One over Diffusivity factor");

    s.append_value(alf.opacities.ny, "/alf_ny", "-", "number of weights in bins");

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

    return true;
}
//***************************************************************************************************

bool two_streams_radiative_transfer::store(const ESP& esp, storage& s) {
    std::shared_ptr<double[]> F_net_h = F_net.get_host_data();
    s.append_table(F_net_h.get(), F_net.get_size(), "/F_net", "W m^-2", "Net Flux");

    std::shared_ptr<double[]> Qheat_h = Qheat.get_host_data();
    s.append_table(Qheat_h.get(), Qheat.get_size(), "/Alf_Qheat", "W m^-3", "Alfrodull Qheat");

    std::shared_ptr<double[]> F_up_tot_h = F_up_tot.get_host_data();
    s.append_table(
        F_up_tot_h.get(), F_up_tot.get_size(), "/F_up_tot", "W m^-2", "Total upward flux");

    std::shared_ptr<double[]> F_down_tot_h = F_down_tot.get_host_data();
    s.append_table(
        F_down_tot_h.get(), F_down_tot.get_size(), "/F_down_tot", "W m^-2", "Total downward flux");

    std::shared_ptr<double[]> F_dir_tot_h = F_dir_tot.get_host_data();
    s.append_table(
        F_dir_tot_h.get(), F_dir_tot.get_size(), "/F_dir_tot", "W m^-2", "Total beam flux");


    std::shared_ptr<double[]> F_up_TOA_spectrum_h = F_up_TOA_spectrum.get_host_data();
    s.append_table(F_up_TOA_spectrum_h.get(),
                   F_up_TOA_spectrum.get_size(),
                   "/F_up_TOA_spectrum",
                   "W m^-2",
                   "Upward Flux per bin at TOA");

    std::shared_ptr<double[]> lambda_wave_h = alf.opacities.dev_opac_wave.get_host_data();
    s.append_table(lambda_wave_h.get(),
                   alf.opacities.dev_opac_wave.get_size(),
                   "/lambda_wave",
                   "m",
                   "Center wavelength");

    std::shared_ptr<double[]> lambda_interwave_h = alf.opacities.dev_opac_interwave.get_host_data();
    s.append_table(lambda_interwave_h.get(),
                   alf.opacities.dev_opac_interwave.get_size(),
                   "/lambda_interwave",
                   "m",
                   "Interface wavelength");

    std::shared_ptr<double[]> lambda_deltawave_h = alf.opacities.dev_opac_deltawave.get_host_data();
    s.append_table(lambda_deltawave_h.get(),
                   alf.opacities.dev_opac_deltawave.get_size(),
                   "/lambda_deltawave",
                   "m",
                   "Wavelength width of bins");

    s.append_value(qheat_scaling, "/qheat_scaling", "-", "QHeat scaling");

    return true;
}


bool two_streams_radiative_transfer::free_memory() {

    return true;
}

// ***************************************************************************************************************
void two_streams_radiative_transfer::print_weighted_band_data_to_file(
    ESP&                        esp,
    int                         nstep,
    int                         column_idx,
    cuda_device_memory<double>& array,
    string                      output_file_base) {
    int nbin = alf.opacities.nbin;
    int ny   = alf.opacities.ny;

    std::string DBG_OUTPUT_DIR = esp.get_output_dir()
                                 + "/alfprof/"
                                   "step_"
                                 + std::to_string(nstep) + "/column_" + std::to_string(column_idx)
                                 + "/";
    create_output_dir(DBG_OUTPUT_DIR);

    // Print out single scattering albedo data

    int                       num_val = array.get_size() / (nbin * ny);
    std::shared_ptr<double[]> array_h =
        integrate_band(*array, *alf.gauss_weights, num_val, nbin, ny);


    string output_file_name = DBG_OUTPUT_DIR + output_file_base + ".dat";

    FILE* output_file = fopen(output_file_name.c_str(), "w");
    // std::shared_ptr<double[]> opac_wg_lay_h =
    //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

    std::shared_ptr<double[]> delta_lambda_h =
        get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

    fprintf(output_file, "bin\t");
    fprintf(output_file, "deltalambda\t");
    for (int i = 0; i < esp.nv; i++)
        fprintf(output_file, "layer[%d]\t", i);
    fprintf(output_file, "\n");

    for (int b = 0; b < nbin; b++) {
        fprintf(output_file, "%d\t", b);
        fprintf(output_file, "%#.6g\t", delta_lambda_h[b]);
        for (int i = 0; i < esp.nv; i++) {
            fprintf(output_file, "%#.6g\t", array_h[b + i * nbin]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}

// ***************************************************************************************************************
// Helper function to print out all datasets for debugging and comparisong to HELIOS
void two_streams_radiative_transfer::debug_print_columns(ESP&   esp,
                                                         double cmustar,
                                                         int    nstep,
                                                         int    column_idx) {
    int nbin = alf.opacities.nbin;
    int ny   = alf.opacities.ny;

    std::string DBG_OUTPUT_DIR = esp.get_output_dir()
                                 + "/alfprof/"
                                   "step_"
                                 + std::to_string(nstep) + "/column_" + std::to_string(column_idx)
                                 + "/";
    create_output_dir(DBG_OUTPUT_DIR);

    {


        double lon = esp.lonlat_h[column_idx * 2 + 0] * 180 / M_PI;
        double lat = esp.lonlat_h[column_idx * 2 + 1] * 180 / M_PI;


        // Print out initial TP profile
        string output_file_name = DBG_OUTPUT_DIR + "tprofile_interp.dat";

        FILE*  tp_output_file = fopen(output_file_name.c_str(), "w");
        string comment = "# Helios TP profile table at lat: [" + std::to_string(lon) + "] lon: ["
                         + std::to_string(lat) + "] mustar: [" + std::to_string(cmustar) + "]\n";

        fprintf(tp_output_file, comment.c_str());
        fprintf(tp_output_file, "#\tT[K]\n");


        std::shared_ptr<double[]> temperature_h = get_cuda_data(*temperature_lay, esp.nv + 1);

        fprintf(tp_output_file, "BOA\t%#.6g\n", temperature_h[esp.nv]);
        for (int i = 0; i < esp.nv; i++) {
            fprintf(tp_output_file, "%d\t%#.6g\n", i, temperature_h[i]);
        }

        fclose(tp_output_file);
    }


    {

        // Print out planck data

        string output_file_name = DBG_OUTPUT_DIR + "plkprofile.dat";

        FILE*                     planck_output_file = fopen(output_file_name.c_str(), "w");
        std::shared_ptr<double[]> planck_h =
            get_cuda_data(*alf.planckband_lay, (esp.nv + 2) * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(planck_output_file, "bin\t");
        fprintf(planck_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nv; i++)
            fprintf(planck_output_file, "layer[%d]\t", i);
        fprintf(planck_output_file, "layer[TOA]\t");
        fprintf(planck_output_file, "layer[BOA]\t");
        fprintf(planck_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(planck_output_file, "%d\t", b);
            fprintf(planck_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nv + 2; i++) {
                fprintf(planck_output_file, "%#.6g\t", planck_h[b * (esp.nv + 2) + i]);
            }
            fprintf(planck_output_file, "\n");
        }
        fclose(planck_output_file);
    }

    {
        // Print out mean molecular weight data

        string output_file_name        = DBG_OUTPUT_DIR + "meanmolmassprofile.dat";
        FILE*  meanmolmass_output_file = fopen(output_file_name.c_str(), "w");

        std::shared_ptr<double[]> meanmolmass_h = get_cuda_data(*alf.meanmolmass_lay, (esp.nv));

        fprintf(meanmolmass_output_file, "layer\t");
        fprintf(meanmolmass_output_file, "meanmolmass\n");

        for (int i = 0; i < esp.nv; i++) {
            fprintf(meanmolmass_output_file, "%d\t", i);
            fprintf(meanmolmass_output_file, "%#.6g\n", meanmolmass_h[i] / AMU);
        }

        fclose(meanmolmass_output_file);
    }

    {
        // Print out mean molecular weight data

        string output_file_name = DBG_OUTPUT_DIR + "deltacolmassprofile.dat";

        FILE*                     deltacolmass_output_file = fopen(output_file_name.c_str(), "w");
        std::shared_ptr<double[]> deltacolmass_h = get_cuda_data(*alf.delta_col_mass, (esp.nv));

        fprintf(deltacolmass_output_file, "layer\t");
        fprintf(deltacolmass_output_file, "delta_col_mass\n");

        for (int i = 0; i < esp.nv; i++) {
            fprintf(deltacolmass_output_file, "%d\t", i);
            fprintf(deltacolmass_output_file, "%#.6g\n", deltacolmass_h[i]);
        }

        fclose(deltacolmass_output_file);
    }

    {

        // Print out opacities data

        int                       num_val = alf.opac_wg_lay.get_size() / (nbin * ny);
        std::shared_ptr<double[]> opac_h =
            integrate_band(*alf.opac_wg_lay, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "opacprofile.dat";

        FILE* opac_output_file = fopen(output_file_name.c_str(), "w");

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(opac_output_file, "bin\t");
        fprintf(opac_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nv; i++)
            fprintf(opac_output_file, "layer[%d]\t", i);
        fprintf(opac_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(opac_output_file, "%d\t", b);
            fprintf(opac_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nv; i++) {
                fprintf(opac_output_file, "%#.6g\t", opac_h[b + i * nbin]);
            }
            fprintf(opac_output_file, "\n");
        }
        fclose(opac_output_file);
    }

    {

        // Print out optical depth data

        int                       num_val = alf.delta_tau_wg.get_size() / (nbin * ny);
        std::shared_ptr<double[]> delta_tau_h =
            integrate_band(*alf.delta_tau_wg, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "opt_depthprofile.dat";

        FILE* opt_depth_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(opt_depth_output_file, "bin\t");
        fprintf(opt_depth_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nv; i++)
            fprintf(opt_depth_output_file, "layer[%d]\t", i);
        fprintf(opt_depth_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(opt_depth_output_file, "%d\t", b);
            fprintf(opt_depth_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nv; i++) {
                fprintf(opt_depth_output_file, "%#.6g\t", delta_tau_h[b + i * nbin]);
            }
            fprintf(opt_depth_output_file, "\n");
        }
        fclose(opt_depth_output_file);
    }

    if (iso) {
        // Print out transmission data

        int                       num_val = alf.trans_wg.get_size() / (nbin * ny);
        std::shared_ptr<double[]> trans_h =
            integrate_band(*alf.trans_wg, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "trans_band_profile.dat";

        FILE* trans_band_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(trans_band_output_file, "bin\t");
        fprintf(trans_band_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nv; i++)
            fprintf(trans_band_output_file, "layer[%d]\t", i);
        fprintf(trans_band_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(trans_band_output_file, "%d\t", b);
            fprintf(trans_band_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nv; i++) {
                fprintf(trans_band_output_file, "%#.6g\t", trans_h[b + i * nbin]);
            }
            fprintf(trans_band_output_file, "\n");
        }
        fclose(trans_band_output_file);
    }
    else {
        {
            // Print out transmission data

            int                       num_val = alf.trans_wg_upper.get_size() / (nbin * ny);
            std::shared_ptr<double[]> trans_h =
                integrate_band(*alf.trans_wg_upper, *alf.gauss_weights, num_val, nbin, ny);


            string output_file_name = DBG_OUTPUT_DIR + "trans_band_upper_profile.dat";

            FILE* trans_band_output_file = fopen(output_file_name.c_str(), "w");
            // std::shared_ptr<double[]> opac_wg_lay_h =
            //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

            std::shared_ptr<double[]> delta_lambda_h =
                get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

            fprintf(trans_band_output_file, "bin\t");
            fprintf(trans_band_output_file, "deltalambda\t");
            for (int i = 0; i < esp.nv; i++)
                fprintf(trans_band_output_file, "layer[%d]\t", i);
            fprintf(trans_band_output_file, "\n");

            for (int b = 0; b < nbin; b++) {
                fprintf(trans_band_output_file, "%d\t", b);
                fprintf(trans_band_output_file, "%#.6g\t", delta_lambda_h[b]);
                for (int i = 0; i < esp.nv; i++) {
                    fprintf(trans_band_output_file, "%#.6g\t", trans_h[b + i * nbin]);
                }
                fprintf(trans_band_output_file, "\n");
            }
            fclose(trans_band_output_file);
        }

        {
            // Print out transmission data

            int                       num_val = alf.trans_wg_lower.get_size() / (nbin * ny);
            std::shared_ptr<double[]> trans_h =
                integrate_band(*alf.trans_wg_lower, *alf.gauss_weights, num_val, nbin, ny);


            string output_file_name = DBG_OUTPUT_DIR + "trans_band_lower_profile.dat";

            FILE* trans_band_output_file = fopen(output_file_name.c_str(), "w");
            // std::shared_ptr<double[]> opac_wg_lay_h =
            //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

            std::shared_ptr<double[]> delta_lambda_h =
                get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

            fprintf(trans_band_output_file, "bin\t");
            fprintf(trans_band_output_file, "deltalambda\t");
            for (int i = 0; i < esp.nv; i++)
                fprintf(trans_band_output_file, "layer[%d]\t", i);
            fprintf(trans_band_output_file, "\n");

            for (int b = 0; b < nbin; b++) {
                fprintf(trans_band_output_file, "%d\t", b);
                fprintf(trans_band_output_file, "%#.6g\t", delta_lambda_h[b]);
                for (int i = 0; i < esp.nv; i++) {
                    fprintf(trans_band_output_file, "%#.6g\t", trans_h[b + i * nbin]);
                }
                fprintf(trans_band_output_file, "\n");
            }
            fclose(trans_band_output_file);
        }
    }

    if (iso) {
        // Print out single scattering albedo data

        int                       num_val = alf.w_0.get_size() / (nbin * ny);
        std::shared_ptr<double[]> w0_h =
            integrate_band(*alf.w_0, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "single_scat_band_profile.dat";

        FILE* singscat_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(singscat_output_file, "bin\t");
        fprintf(singscat_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nv; i++)
            fprintf(singscat_output_file, "layer[%d]\t", i);
        fprintf(singscat_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(singscat_output_file, "%d\t", b);
            fprintf(singscat_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nv; i++) {
                fprintf(singscat_output_file, "%#.6g\t", w0_h[b + i * nbin]);
            }
            fprintf(singscat_output_file, "\n");
        }
        fclose(singscat_output_file);
    }
    else {
        {
            // Print out single scattering albedo data

            int                       num_val = alf.w_0_upper.get_size() / (nbin * ny);
            std::shared_ptr<double[]> w0_h =
                integrate_band(*alf.w_0_upper, *alf.gauss_weights, num_val, nbin, ny);


            string output_file_name = DBG_OUTPUT_DIR + "single_scat_band_upper_profile.dat";

            FILE* singscat_output_file = fopen(output_file_name.c_str(), "w");
            // std::shared_ptr<double[]> opac_wg_lay_h =
            //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

            std::shared_ptr<double[]> delta_lambda_h =
                get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

            fprintf(singscat_output_file, "bin\t");
            fprintf(singscat_output_file, "deltalambda\t");
            for (int i = 0; i < esp.nv; i++)
                fprintf(singscat_output_file, "layer[%d]\t", i);
            fprintf(singscat_output_file, "\n");

            for (int b = 0; b < nbin; b++) {
                fprintf(singscat_output_file, "%d\t", b);
                fprintf(singscat_output_file, "%#.6g\t", delta_lambda_h[b]);
                for (int i = 0; i < esp.nv; i++) {
                    fprintf(singscat_output_file, "%#.6g\t", w0_h[b + i * nbin]);
                }
                fprintf(singscat_output_file, "\n");
            }
            fclose(singscat_output_file);
        }

        {
            // Print out single scattering albedo data

            int                       num_val = alf.w_0_lower.get_size() / (nbin * ny);
            std::shared_ptr<double[]> w0_h =
                integrate_band(*alf.w_0_lower, *alf.gauss_weights, num_val, nbin, ny);


            string output_file_name = DBG_OUTPUT_DIR + "single_scat_band_lower_profile.dat";

            FILE* singscat_output_file = fopen(output_file_name.c_str(), "w");
            // std::shared_ptr<double[]> opac_wg_lay_h =
            //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

            std::shared_ptr<double[]> delta_lambda_h =
                get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

            fprintf(singscat_output_file, "bin\t");
            fprintf(singscat_output_file, "deltalambda\t");
            for (int i = 0; i < esp.nv; i++)
                fprintf(singscat_output_file, "layer[%d]\t", i);
            fprintf(singscat_output_file, "\n");

            for (int b = 0; b < nbin; b++) {
                fprintf(singscat_output_file, "%d\t", b);
                fprintf(singscat_output_file, "%#.6g\t", delta_lambda_h[b]);
                for (int i = 0; i < esp.nv; i++) {
                    fprintf(singscat_output_file, "%#.6g\t", w0_h[b + i * nbin]);
                }
                fprintf(singscat_output_file, "\n");
            }
            fclose(singscat_output_file);
        }
    }
    //***********************************************************************************************
    if (iso) {
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.M_term, "M_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.N_term, "N_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.P_term, "P_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.G_plus, "G_plus_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.G_minus, "G_minus_profile");
    }
    else {
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.M_upper, "M_upper_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.M_lower, "M_lower_profile");

        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.N_upper, "N_upper_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.N_lower, "N_lower_profile");

        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.P_upper, "P_upper_profile");
        print_weighted_band_data_to_file(esp, nstep, column_idx, alf.P_lower, "P_lower_profile");

        print_weighted_band_data_to_file(
            esp, nstep, column_idx, alf.G_plus_upper, "G_plus_upper_profile");
        print_weighted_band_data_to_file(
            esp, nstep, column_idx, alf.G_plus_lower, "G_plus_lower_profile");

        print_weighted_band_data_to_file(
            esp, nstep, column_idx, alf.G_minus_upper, "G_minus_upper_profile");
        print_weighted_band_data_to_file(
            esp, nstep, column_idx, alf.G_minus_lower, "G_minus_lower_profile");
    }
    //***********************************************************************************************
    {
        // Print out downward flux

        int                       num_val = F_down_wg.get_size() / (nbin * ny);
        std::shared_ptr<double[]> fd_h =
            integrate_band(*F_down_wg, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "F_down_profile.dat";

        FILE* Fd_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(Fd_output_file, "bin\t");
        fprintf(Fd_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nvi; i++)
            fprintf(Fd_output_file, "interface[%d]\t", i);
        fprintf(Fd_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(Fd_output_file, "%d\t", b);
            fprintf(Fd_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nvi; i++) {
                fprintf(Fd_output_file, "%#.6g\t", fd_h[b + i * nbin]);
            }
            fprintf(Fd_output_file, "\n");
        }
        fclose(Fd_output_file);
    }

    {
        // Print out downward flux

        int                       num_val = F_up_wg.get_size() / (nbin * ny);
        std::shared_ptr<double[]> fu_h =
            integrate_band(*F_up_wg, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "F_up_profile.dat";

        FILE* Fu_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(Fu_output_file, "bin\t");
        fprintf(Fu_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nvi; i++)
            fprintf(Fu_output_file, "interface[%d]\t", i);
        fprintf(Fu_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(Fu_output_file, "%d\t", b);
            fprintf(Fu_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nvi; i++) {
                fprintf(Fu_output_file, "%#.6g\t", fu_h[b + i * nbin]);
            }
            fprintf(Fu_output_file, "\n");
        }
        fclose(Fu_output_file);
    }

    {
        // Print out direct beam flux

        int                       num_val = F_dir_wg.get_size() / (nbin * ny);
        std::shared_ptr<double[]> fdir_h =
            integrate_band(*F_dir_wg, *alf.gauss_weights, num_val, nbin, ny);


        string output_file_name = DBG_OUTPUT_DIR + "F_dir_profile.dat";

        FILE* Fdir_output_file = fopen(output_file_name.c_str(), "w");
        // std::shared_ptr<double[]> opac_wg_lay_h =
        //     get_cuda_data(*alf.opac_wg_lay, esp.nv * nbin);

        std::shared_ptr<double[]> delta_lambda_h =
            get_cuda_data(*alf.opacities.dev_opac_deltawave, nbin);

        fprintf(Fdir_output_file, "bin\t");
        fprintf(Fdir_output_file, "deltalambda\t");
        for (int i = 0; i < esp.nvi; i++)
            fprintf(Fdir_output_file, "interface[%d]\t", i);
        fprintf(Fdir_output_file, "\n");

        for (int b = 0; b < nbin; b++) {
            fprintf(Fdir_output_file, "%d\t", b);
            fprintf(Fdir_output_file, "%#.6g\t", delta_lambda_h[b]);
            for (int i = 0; i < esp.nvi; i++) {
                fprintf(Fdir_output_file, "%#.6g\t", fdir_h[b + i * nbin]);
            }
            fprintf(Fdir_output_file, "\n");
        }
        fclose(Fdir_output_file);
    }


    {
        // Print out alf qheat

        int num_val = F_dir_wg.get_size() / (nbin * ny);


        int col_offset = column_idx * esp.nv;

        string output_file_name = DBG_OUTPUT_DIR + "alf_qheat_profile.dat";

        FILE* output_file = fopen(output_file_name.c_str(), "w");

        std::shared_ptr<double[]> qh_h = get_cuda_data(&((Qheat.ptr())[col_offset]), esp.nv);

        fprintf(output_file, "level\tqheat\n");

        for (int i = 0; i < esp.nv; i++) {
            fprintf(output_file, "%d\t%#.6g\n", i, qh_h[i]);
        }

        fclose(output_file);
    }
}
