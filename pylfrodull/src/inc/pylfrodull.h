
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
			  long dev_p_lay, // io, mmm, kil
			  long dev_p_int, // ioi, mmmi, kii
			  long dev_opac_wg_lay, // io
			  long dev_opac_wg_int, // ioi
			  long dev_meanmolmass_lay, // mmm
			  long dev_meanmolmass_int, // mmmi
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
			       );
