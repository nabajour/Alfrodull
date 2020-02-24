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
#include "phy_module_base.h"
#include "cudaDeviceMemory.h"

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

    bool configure(config_file &config_reader);

    virtual bool free_memory();

    int    nlayer;
    bool   iso;
    double T_star;
    bool   real_star;
    double fake_opac;
    double T_surf;
    double albedo;
    double g_0;
    double epsi;
    double mu_star; // not a config
    bool   scat;
    bool   scat_corr;
    double R_planet;
    double R_star;
    double a; // ?
    bool   dir_beam;
    bool   geom_zenith_corr;
    double f_factor; // still needed?
    double w_0_limit;

    double i2s_transition;

    string opacities_file;

  void print_config();


private:
  alfrodull_engine alf;

  cuda_device_memory<double> pressure_int;
  cuda_device_memory<double> temperature_int;
  
  cuda_device_memory<double> F_down_wg;
  cuda_device_memory<double> F_up_wg;
  cuda_device_memory<double> Fc_down_wg;
  cuda_device_memory<double> Fc_up_wg;
  cuda_device_memory<double> F_dir_wg;
  cuda_device_memory<double> Fc_dir_wg;
  cuda_device_memory<double> F_down_tot;
  cuda_device_memory<double> F_up_tot;
  cuda_device_memory<double> F_down_band;
  cuda_device_memory<double> F_up_band;
  cuda_device_memory<double> F_dir_band;
  
  cuda_device_memory<double> F_net;
  cuda_device_memory<double> F_net_diff;

  cuda_device_memory<double> star_flux;  
};
