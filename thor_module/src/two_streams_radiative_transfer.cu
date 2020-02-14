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

two_streams_radiative_transfer::two_streams_radiative_transfer() {
}

two_streams_radiative_transfer::~two_streams_radiative_transfer() {
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

    // need frequency bins
    // initialise opacities table -> gives frequency bins
    // initialise planck tables
    // initialise alf

    // TODO: understand what needs to be stored per column. and what can be global for internal conputation
    // what needs to be passed outside or stored should be global, others can be per column

    return true;
}

bool two_streams_radiative_transfer::initial_conditions(const ESP&             esp,
                                                        const SimulationSetup& sim,
                                                        storage*               s) {
    // what s hould be initialised here and what is to initialise at each loop ?
    // what to initialise here and what to do in initialise memory ?
    return true;
}

bool two_streams_radiative_transfer::phy_loop(ESP&                   esp,
                                              const SimulationSetup& sim,
                                              int                    nstep, // Step number
                                              double                 time_step)             // Time-step [s]
{
    // TODO: check - those seem to be preset in the new version,
    // but probably only as DeltaP/g
    // (long)*delta_col_mass,
    //   (long)*delta_col_upper,
    //   (long)*delta_col_lower,


    // Check in here, some values from initial setup might change per column: e.g. mu_star;
    compute_radiative_transfer(
        double* dev_starflux, // in: pil
        double*
                    dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
        double*     dev_T_int,                 // in: it, pii, ioi, mmmi, kii
        double*     dev_p_lay,                 // in: io, mmm, kil
        double*     dev_p_int,                 // in: ioi, mmmi, kii
        const bool& interp_and_calc_flux_step, // done at each step
        double*     z_lay,
        bool        single_walk, // (?) - TODO: check
        double*     F_down_wg,
        double*     F_up_wg,
        double*     Fc_down_wg,
        double*     Fc_up_wg,
        double*     F_dir_wg,
        double*     Fc_dir_wg,
        double      delta_tau_limit,
        double*     F_down_tot,
        double*     F_up_tot,
        double*     F_net,
        double*     F_down_band,
        double*     F_up_band,
        double*     F_dir_band,
        double*     gauss_weight);


    return true;
}

bool two_streams_radiative_transfer::store(const ESP& esp, storage& s) {

    return true;
}


bool two_streams_radiative_transfer::free_memory() {

    return true;
}
