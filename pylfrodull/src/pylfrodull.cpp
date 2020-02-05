#include "pylfrodull.h"
#include "alfrodullib.h"

#include <pybind11/pybind11.h>
#include <functional>

namespace py = pybind11;

py::function fn_clbck;

void call_callback()
{
  printf("Called call_callback\n");
  fn_clbck();
  printf("Came back from callback\n");
  fn_clbck.release();
}

std::function<void()> calc_z_callback = call_callback;

void set_callback(py::object & fn)
{
  printf("Called set_callback\n");
  fn_clbck = fn;

  
  set_z_calc_function(calc_z_callback);

}





PYBIND11_MODULE(pylfrodull, m) {


  
    m.doc() = "python wrapper for Alfrodull"; // optional module docstring

    m.def("integrate_flux", &wrap_integrate_flux, "Integrate the flux");
    m.def("pyprepare_compute_flux", &wrap_prepare_compute_flux, "prepare computation of fluxes");
    m.def("pycompute_transmission_iso", &wrap_calculate_transmission_iso, "compute transmission iso");
    m.def("pycompute_transmission_noniso", &wrap_calculate_transmission_noniso, "compute transmission noniso");
    m.def("pycompute_direct_beam_flux", &wrap_direct_beam_flux, "compute direct beam flux");
    m.def("pypopulate_spectral_flux_iso", &wrap_populate_spectral_flux_iso, "populate spectral flux iso");
    m.def("pypopulate_spectral_flux_noniso", &wrap_populate_spectral_flux_noniso, "populate spectral flux noniso");
    m.def("pycompute_radiative_transfer", &wrap_compute_radiative_transfer, "compute radiative transfer");
    m.def("init_alfrodull", &init_alfrodull, "initialise Alfrodull Engine");
    m.def("deinit_alfrodull", &deinit_alfrodull, "deinitialise Alfrodull Engine");
    m.def("init_parameters", &init_parameters, "initialise global sim parameters");
    m.def("allocate", &allocate, "allocate internal memory");
    m.def("get_dev_pointers", &get_device_pointers_for_helios_write, "Get device pointers");
    m.def("prepare_planck_table", &prepare_planck_table, "Prepare planck table");
    m.def("correct_incident_energy", &correct_incident_energy, "Correct incident flux");

    m.def("set_callback", &set_callback, "Set function callback");
    m.def("call_callback", &call_callback, "call function callback");
}
