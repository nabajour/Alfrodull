#include "pylfrodull.h"
#include "alfrodullib.h"

#include <pybind11/pybind11.h>




PYBIND11_MODULE(pylfrodull, m) {
    m.doc() = "python wrapper for Alfrodull"; // optional module docstring

    m.def("integrate_flux", &wrap_integrate_flux, "Integrate the flux");
    m.def("pyprepare_compute_flux", &wrap_prepare_compute_flux, "prepare computation of fluxes");
    m.def("pycompute_transmission_iso", &wrap_calculate_transmission_iso, "compute transmission iso");
    m.def("pycompute_transmission_noniso", &wrap_calculate_transmission_noniso, "compute transmission noniso");
    m.def("pycompute_direct_beam_flux", &wrap_direct_beam_flux, "compute direct beam flux");
    m.def("pypopulate_spectral_flux_iso", &wrap_populate_spectral_flux_iso, "populate spectral flux iso");
    m.def("pypopulate_spectral_flux_noniso", &wrap_populate_spectral_flux_noniso, "populate spectral flux noniso");
    m.def("init_alfrodull", &init_alfrodull, "initialise Alfrodull Engine");
    m.def("deinit_alfrodull", &deinit_alfrodull, "deinitialise Alfrodull Engine");
    m.def("init_parameters", &init_parameters, "initialise global sim parameters");
    m.def("allocate", &allocate, "allocate internal memory");
    m.def("get_dev_pointers", &get_device_pointers_for_helios_write, "Get device pointers");
    m.def("prepare_planck_table", &prepare_planck_table, "Prepare planck table");
}
