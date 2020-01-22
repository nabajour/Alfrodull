#include "pylfrodull.h"
#include "alfrodullib.h"


#include <pybind11/pybind11.h>


PYBIND11_MODULE(pylfrodull, m) {
    m.doc() = "python wrapper for Alfrodull"; // optional module docstring

    m.def("integrate_flux", &wrap_integrate_flux, "Integrate the flux");
    m.def("prepare_compute_flux", &wrap_prepare_compute_flux, "prepare computation of fluxes");
    
}
