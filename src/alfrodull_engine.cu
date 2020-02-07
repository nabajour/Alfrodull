#include "alfrodull_engine.h"

alfrodull_engine::alfrodull_engine()
{
  printf("Creating Alfrodull engine\n");
}

void alfrodull_engine::load_opacities(const string & filename)
{
  printf("Loading opacities from %s\n", filename.c_str());

  opacities.load_opacity_table(filename);
}

void alfrodull_engine::init()
{
  printf("Alfrodull Init\n");

  load_opacities("input/opac_sample.h5");
}

void alfrodull_engine::set_parameters(const int & nlayer_,
				      const bool & iso_,
				      const double & T_star_)
{
  nlayer = nlayer_;
  ninterface = nlayer + 1;
  iso = iso_;
  T_star = T_star_;
  
  // TODO: maybe should stay in opacities object
  nbin = opacities.nbin;

  // prepare_planck_table();
}

void alfrodull_engine::allocate_internal_variables()
{
  int nlayer_nbin = nlayer*opacities.nbin;
  int nlayer_plus2_nbin = (nlayer + 2)*opacities.nbin;
  int ninterface_nbin = ninterface*opacities.nbin;
  int nlayer_wg_nbin = nlayer*opacities.ny*opacities.nbin;
  int ninterface_wg_nbin = ninterface*opacities.ny*opacities.nbin;

  // scatter cross section layer and interface
  // those are shared for print out
  printf("nlayer_nbin %d\n", nlayer_nbin);
  printf("nlayer_plus2_nbin %d\n", nlayer_plus2_nbin);
  printf("ninterface_nbin %d\n", ninterface_nbin);
  printf("nlayer_wg_nbin %d\n", nlayer_wg_nbin);
  printf("ninterface_wg_nbin %d\n", ninterface_wg_nbin);
  scatter_cross_section_lay.allocate(nlayer_nbin);
  scatter_cross_section_inter.allocate(ninterface_nbin);
  planckband_lay.allocate(nlayer_plus2_nbin);
  planckband_int.allocate(ninterface_nbin);

  
  if (iso)
    {
      delta_tau_wg.allocate(nlayer_wg_nbin);
    }
  else
    {
      delta_tau_wg_upper.allocate(nlayer_wg_nbin);
      delta_tau_wg_lower.allocate(nlayer_wg_nbin);
    }
  
  // flux computation internal quantities
  // TODO: not needed to allocate everything, depending on iso or noniso
  if (iso)
    { 
      M_term.allocate(nlayer_wg_nbin);
      N_term.allocate(nlayer_wg_nbin);   
      P_term.allocate(nlayer_wg_nbin);
      G_plus.allocate(nlayer_wg_nbin);
      G_minus.allocate(nlayer_wg_nbin);
      w_0.allocate(nlayer_wg_nbin);
    }
  else
    { 
      M_upper.allocate(nlayer_wg_nbin);
      M_lower.allocate(nlayer_wg_nbin);
      N_upper.allocate(nlayer_wg_nbin);
      N_lower.allocate(nlayer_wg_nbin);
      P_upper.allocate(nlayer_wg_nbin);
      P_lower.allocate(nlayer_wg_nbin);
      G_plus_upper.allocate(nlayer_wg_nbin);
      G_plus_lower.allocate(nlayer_wg_nbin);
      G_minus_upper.allocate(nlayer_wg_nbin);
      G_minus_lower.allocate(nlayer_wg_nbin);
      w_0_upper.allocate(nlayer_wg_nbin);
      w_0_lower.allocate(nlayer_wg_nbin);
    }

  //  dev_T_int.allocate(ninterface);

  // column mass
  // TODO: computed by grid in helios, should be computed by alfrodull or comes from THOR?
  delta_col_mass.allocate(nlayer);
  delta_col_upper.allocate(nlayer);
  delta_col_lower.allocate(nlayer);
  

  
  // dev_meanmolmass_int.allocate(ninterface);
  
  // dev_opac_wg_lay.allocate(nlayer_wg_nbin);

  // dev_trans_wg.allocate(nlayer_wg_nbin);

  // if (!iso)
  //   {
  //       dev_opac_wg_int.allocate(ninterface_wg_nbin);
  // 	dev_trans_wg_upper.allocate(nlayer_wg_nbin);
  // 	dev_trans_wg_lower.allocate(nlayer_wg_nbin);
  //   }
  
}

// return device pointers for helios data save
// TODO: how ugly can it get, really?
std::tuple<long, long, long,
	   long, long, long,
	   long, long, long,
	   long, long, long,
	   long, int, int>
alfrodull_engine::get_device_pointers_for_helios_write( )
{
  printf("Mem pointers:\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%d\n%d\n",
	 *scatter_cross_section_lay,
	 *scatter_cross_section_inter,
	 *opacities.dev_opac_interwave,
	 *opacities.dev_opac_deltawave,
	 *planckband_lay,
	 *planckband_int,
	 *plancktable.planck_grid,
	 *delta_tau_wg,
	 *delta_tau_wg_upper,
	 *delta_tau_wg_lower,
	 *delta_col_mass,
	 *delta_col_upper,
	 *delta_col_lower,
	 plancktable.dim,
	 plancktable.step);
	 
  return std::make_tuple((long) *scatter_cross_section_lay,
			 (long) *scatter_cross_section_inter,
			 (long) *opacities.dev_opac_interwave,
			 (long) *opacities.dev_opac_deltawave,
			 (long) *planckband_lay,
			 (long) *planckband_int,
			 (long) *plancktable.planck_grid,
			 (long) *delta_tau_wg,
			 (long) *delta_tau_wg_upper,
			 (long) *delta_tau_wg_lower,
			 (long) *delta_col_mass,
			 (long) *delta_col_upper,
			 (long) *delta_col_lower,
			 plancktable.dim,
			 plancktable.step);
}

// TODO: check how to enforce this: must be called after loading opacities and setting parameters
void alfrodull_engine::prepare_planck_table()
{
  plancktable.construct_planck_table(*opacities.dev_opac_interwave,
				     *opacities.dev_opac_deltawave,
				     opacities.nbin,
				     T_star);
				     
}

void alfrodull_engine::correct_incident_energy(double * starflux_array_ptr,
					       bool real_star,
					       bool energy_budget_correction)
{
  printf("T_star %g, energy budget_correction: %s\n", T_star, energy_budget_correction?"true":"false" );
  if (T_star > 10 && energy_budget_correction)
    {
      dim3 grid((int(opacities.nbin) + 15 )/16, 1, 1 );
      dim3 block(16,1,1);
      
      corr_inc_energy<<<grid, block>>>(*plancktable.planck_grid,
		      starflux_array_ptr,
		      *opacities.dev_opac_deltawave,
		      real_star,
		      opacities.nbin,
		      T_star,
		      plancktable.dim);
      
      cudaDeviceSynchronize();

      
    }

  // //nplanck_grid = (plancktable.dim+1)*opacities.nbin;
  // // print out planck grid for debug
  // std::unique_ptr<double[]> plgrd = std::make_unique<double[]>(plancktable.nplanck_grid);
  
  // plancktable.planck_grid.fetch(plgrd);
  // for (int i = 0; i < plancktable.nplanck_grid; i++)
  //   printf("array[%d] : %g\n", i, plgrd[i]);
}


void alfrodull_engine::set_z_calc_func( std::function<void()> & fun)
{
  calc_z_func = fun;
}

void alfrodull_engine::call_z_callback()
{
  if (calc_z_func)
    calc_z_func();
  
}
