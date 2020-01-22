#include "pylfrodull.h"
#include "alfrodullib.h"


#include <pybind11/pybind11.h>


bool wrap_prepare_compute_flux(
			  long dev_planckband_lay,  // csp, cse
			  long dev_planckband_grid,  // pil, pii
			  long dev_planckband_int,  // pii
			  long dev_starflux, // pil
			  long dev_opac_interwave,  // csp
			  long dev_opac_deltawave,  // csp, cse
			  long dev_F_down_tot, // cse
			  long dev_T_lay, // it, pil, io, mmm, kil
			  long dev_T_int, // it, pii, ioi, mmmi, kii
			  long dev_ktemp, // io, mmm, mmmi
			  long dev_p_lay, // io, mmm, kil
			  long dev_p_int, // ioi, mmmi, kii
			  long dev_kpress, // io, mmm, mmmi
			  long dev_opac_k, // io
			  long dev_opac_wg_lay, // io
			  long dev_opac_wg_int, // ioi
			  long dev_opac_scat_cross, // io
			  long dev_scat_cross_lay, // io
			  long dev_scat_cross_int, // ioi
			  long dev_meanmolmass_lay, // mmm
			  long dev_meanmolmass_int, // mmmi
			  long dev_opac_meanmass, // mmm, mmmi
			  long dev_opac_kappa, // kil, kii
			  long dev_entr_temp, // kil, kii
			  long dev_entr_press, // kil, kii
			  long dev_kappa_lay, // kil
			  long dev_kappa_int, // kii
			  const int & ninterface, // it, pii, mmmi, kii
			  const int & nbin, // csp, cse, pil, pii, io
			  const int & nlayer, // csp, cse, pil, io, mmm, kil
			  const int & iter_value, // cse // TODO: check what this is for. Should maybe be external
			  const int & real_star, // pil
			  const int & npress, // io, mmm, mmmi
			  const int & ntemp, // io, mmm, mmmi
			  const int & ny, // io
			  const int & entr_npress, // kii, kil
			  const int & entr_ntemp, // kii, kil		  
			  const double & fake_opac, // io
			  const double & T_surf, // csp, cse, pil
			  const double & surf_albedo, // cse
			  const int & dim, // pil, pii
			  const int & step, // pil, pii
			  const bool & use_kappa_manual, // ki
			  const double & kappa_manual_value, // ki	     
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
				  (double *)dev_opac_interwave,  // csp
				  (double *)dev_opac_deltawave,  // csp, cse
				  (double *)dev_F_down_tot, // cse
				  (double *)dev_T_lay, // it, pil, io, mmm, kil
				  (double *)dev_T_int, // it, pii, ioi, mmmi, kii
				  (double *)dev_ktemp, // io, mmm, mmmi
				  (double *)dev_p_lay, // io, mmm, kil
				  (double *)dev_p_int, // ioi, mmmi, kii
				  (double *)dev_kpress, // io, mmm, mmmi
				  (double *)dev_opac_k, // io
				  (double *)dev_opac_wg_lay, // io
				  (double *)dev_opac_wg_int, // ioi
				  (double *)dev_opac_scat_cross, // io
				  (double *)dev_scat_cross_lay, // io
				  (double *)dev_scat_cross_int, // ioi
				  (double *)dev_meanmolmass_lay, // mmm
				  (double *)dev_meanmolmass_int, // mmmi
				  (double *)dev_opac_meanmass, // mmm, mmmi
				  (double *)dev_opac_kappa, // kil, kii
				  (double *)dev_entr_temp, // kil, kii
				  (double *)dev_entr_press, // kil, kii
				  (double *)dev_kappa_lay, // kil
				  (double *)dev_kappa_int, // kii
				  ninterface, // it, pii, mmmi, kii
				  nbin, // csp, cse, pil, pii, io
				  nlayer, // csp, cse, pil, io, mmm, kil
				  iter_value, // cse // TODO: check what this is for. Should maybe be external
				  real_star, // pil
				  npress, // io, mmm, mmmi
				  ntemp, // io, mmm, mmmi
				  ny, // io
				  entr_npress,
				  entr_ntemp,
				  fake_opac, // io
				  T_surf, // csp, cse, pil
				  surf_albedo, // cse
				  dim, // pil, pii
				  step, // pil, pii
				  use_kappa_manual, // ki
				  kappa_manual_value, // ki	     
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
    
}
