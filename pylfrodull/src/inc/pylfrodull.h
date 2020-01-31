
bool wrap_prepare_compute_flux(
			  long dev_planckband_lay,  // csp, cse
			  long dev_planckband_grid,  // pil, pii
			  long dev_planckband_int,  // pii
			  long dev_starflux, // pil
			  long dev_F_down_tot, // cse
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
			       );
