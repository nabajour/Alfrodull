#include "pylfrodull.h"
#include "alfrodullib.h"


#include <pybind11/pybind11.h>


bool wrap_prepare_compute_flux(
			  long dev_planckband_lay,  // csp, cse
			  long dev_planckband_grid,  // pil, pii
			  long dev_planckband_int,  // pii
			  long dev_starflux, // pil
			  long dev_T_lay, // it, pil, io, mmm, kil
			  long dev_T_int, // it, pii, ioi, mmmi, kii
			  long dev_p_lay, // io, mmm, kil
			  long dev_p_int, // ioi, mmmi, kii
			  long dev_opac_wg_lay, // io
			  long dev_opac_wg_int, // ioi
			  long dev_meanmolmass_lay, // mmm
			  long dev_meanmolmass_int, // mmmi
			  const int & ninterface, // it, pii, mmmi, kii
			  const int & nbin, // csp, cse, pil, pii, io
			  const int & nlayer, // csp, cse, pil, io, mmm, kil
			  const int & real_star, // pil
			  const double & fake_opac, // io
			  const double & T_surf, // csp, cse, pil
			  const double & surf_albedo, // cse
			  const int & dim, // pil, pii
			  const int & step, // pil, pii
			  const bool & iso, // pii
			  const bool & correct_surface_emissions,
			  const bool & interp_and_calc_flux_step
			  
			       )
{

  bool ret = prepare_compute_flux(
				  (double *)dev_planckband_lay,  // csp, cse
				  (double *)dev_planckband_grid,  // pil, pii
				  (double *)dev_planckband_int,  // pii
				  (double *)dev_starflux, // pil
				  (double *)dev_F_down_tot, // cse
				  (double *)dev_T_lay, // it, pil, io, mmm, kil
				  (double *)dev_T_int, // it, pii, ioi, mmmi, kii
				  (double *)dev_p_lay, // io, mmm, kil
				  (double *)dev_p_int, // ioi, mmmi, kii
				  (double *)dev_opac_wg_lay, // io
				  (double *)dev_opac_wg_int, // ioi
				  (double *)dev_meanmolmass_lay, // mmm
				  (double *)dev_meanmolmass_int, // mmmi
				  ninterface, // it, pii, mmmi, kii
				  nbin, // csp, cse, pil, pii, io
				  nlayer, // csp, cse, pil, io, mmm, kil
				  real_star, // pil
				  fake_opac, // io
				  T_surf, // csp, cse, pil
				  surf_albedo, // cse
				  dim, // pil, pii
				  step, // pil, pii
				  iso, // pii
				  correct_surface_emissions,
				  interp_and_calc_flux_step
				  );
    return ret;
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
    m.def("init_alfrodull", &init_alfrodull, "initialise Alfrodull Engine");
    m.def("deinit_alfrodull", &deinit_alfrodull, "deinitialise Alfrodull Engine");
    m.def("init_parameters", &init_parameters, "initialise global sim parameters");
    m.def("allocate", &allocate, "allocate internal memory");
    m.def("get_dev_pointers", &get_device_pointers_for_helios_write, "Get device pointers");
}
