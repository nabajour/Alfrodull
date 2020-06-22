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

#pragma once

#include "alfrodull_engine.h"
#include "alfrodullib.h"
#include "cuda_device_memory.h"
#include "phy_module_base.h"

class two_streams_radiative_transfer : public phy_module_base
{
public:
    two_streams_radiative_transfer();
    ~two_streams_radiative_transfer();

    bool initialise_memory(const ESP &esp, device_RK_array_manager &phy_modules_core_arrays);
    bool initial_conditions(const ESP &esp, const SimulationSetup &sim, storage *s);

    // virtual bool dyn_core_loop_init(const ESP& esp) {
    //     return true;
    // };
    // virtual bool dyn_core_loop_slow_modes(const ESP&             esp,
    //                                       const SimulationSetup& sim,

    //                                       int    nstep, // Step number
    //                                       double times) // Time-step [s]
    // {
    //     return true;
    // };
    // virtual bool dyn_core_loop_fast_modes(const ESP&             esp,
    //                                       const SimulationSetup& sim,
    //                                       int                    nstep, // Step number
    //                                       double                 time_step)             // Time-step [s]
    // {
    //     return true;
    // };
    // virtual bool dyn_core_loop_end(const ESP& esp) {
    //     return true;
    // };

    bool phy_loop(ESP &                  esp,
                  const SimulationSetup &sim,
                  int                    nstep, // Step number
                  double                 time_step);            // Time-step [s]

    bool store(const ESP &esp, storage &s);

    bool store_init(storage &s);

    bool configure(config_file &config_reader);

    virtual bool free_memory();

    int    nlayer;
    bool   iso;
    double T_star;
    double T_internal = 100.0;
    bool   real_star  = false;
    double fake_opac;
    double g_0;
    double epsi        = 1.0 / 2.0;
    double epsilon_2   = 2.0 / 3.0;
    double diffusivity = 2.0;
    double mu_star; // not a config
    bool   scat;
    bool   scat_corr;

    bool store_w0_g0 = false;

    // config
    double R_star_config;           // [R_sun]
    double planet_star_dist_config; // [AU]
    double R_star_SI;               // [m]
    double planet_star_dist_SI;     // [m]


    //    double a; // ?
    bool thomas = false;
    // two debugging variables
    // run in single walk mode
    bool scat_single_walk = true;
    // add offset to opacities
    double experimental_opacities_offset = 0.0;

    bool   dir_beam;
    bool   geom_zenith_corr;
    double w_0_limit;

    double i2s_transition;

    string opacities_file;

    string stellar_spectrum_file;
    void   print_config();

    double F_intern = 0.0;

    double qheat_scaling = 0.0; // scaling of QHeat to slowly ramp up over multiple steps

    int compute_every_n_iteration = 1;

    bool   G_pm_limiter             = true;
    double G_pm_denom_limit         = 1e-5;
    double mu_star_wiggle_increment = 0.001;

    // TODO: check this. if we are starting up and not at iteration 0,
    // we need either to reload the Qheat or something to compute it (net flux) or recompute it.
    // or we'll have discrepancies between loading initial conditions and running cases

    bool start_up = true;

    bool store_weight_flux = true;
    bool store_band_flux   = true;
    bool store_updown_flux = true;
    bool store_net_flux    = true;

    bool   clouds = false;
    double fcloud = 0.5;

    string cloud_filename;


private:
    int spinup_start_step = -1;
    int spinup_stop_step  = -1;

    int spindown_start_step = -1;
    int spindown_stop_step  = -1;

    int              last_step = 0;
    alfrodull_engine alf;

    cuda_device_memory<double> pressure_int;
    cuda_device_memory<double> temperature_int;
    cuda_device_memory<double> temperature_lay;

    // interfadce fluxes
    cuda_device_memory<double> F_down_wg;
    cuda_device_memory<double> F_up_wg;
    // center of layer fluxes
    cuda_device_memory<double> Fc_down_wg;
    cuda_device_memory<double> Fc_up_wg;
    // direct beam, interface and center
    cuda_device_memory<double> F_dir_wg;
    cuda_device_memory<double> Fc_dir_wg;
    // total integrated flux
    cuda_device_memory<double> F_down_tot;
    cuda_device_memory<double> F_up_tot;
    cuda_device_memory<double> F_dir_tot;
    // flux per freq band
    cuda_device_memory<double> F_down_band;
    cuda_device_memory<double> F_up_band;
    cuda_device_memory<double> F_dir_band;

    // Top of atmosphere flux spectrum
    cuda_device_memory<double> F_up_TOA_spectrum;
    cuda_device_memory<double> F_net;
    cuda_device_memory<double> F_net_diff;

    cuda_device_memory<double> star_flux;
    cuda_device_memory<double> g_0_tot_lay;
    cuda_device_memory<double> g_0_tot_int;
    cuda_device_memory<double> cloud_abs_cross_lay;
    cuda_device_memory<double> cloud_abs_cross_int;
    cuda_device_memory<double> cloud_scat_cross_lay;
    cuda_device_memory<double> cloud_scat_cross_int;

    cuda_device_memory<double> Qheat;

    cuda_device_memory<double> w0_tot;
    cuda_device_memory<double> g0_tot;
    // Debug print out function
    void debug_print_columns(ESP &esp, double cmustar, int nstep, int column_idx);
    void print_weighted_band_data_to_file(ESP &                       esp,
                                          int                         nstep,
                                          int                         column_idx,
                                          cuda_device_memory<double> &array,
                                          string                      output_file_base);
};
