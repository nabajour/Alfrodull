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

bool two_streams_radiative_transfer::initialise_memory(
    const ESP &              esp,
    device_RK_array_manager &phy_modules_core_arrays) {
    return true;
}

bool two_streams_radiative_transfer::initial_conditions(const ESP &            esp,
                                                        const SimulationSetup &sim,
                                                        storage *              s) {
    return true;
}

bool two_streams_radiative_transfer::phy_loop(ESP &                  esp,
                                              const SimulationSetup &sim,
                                              int                    nstep, // Step number
                                              double                 time_step)             // Time-step [s]
{

    return true;
}

bool two_streams_radiative_transfer::store(const ESP &esp, storage &s) {

    return true;
}

bool two_streams_radiative_transfer::configure(config_file &config_reader) {

    return true;
}

bool two_streams_radiative_transfer::free_memory() {

    return true;
}
