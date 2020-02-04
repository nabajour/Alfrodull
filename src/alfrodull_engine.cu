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
  scatter_cross_section_lay.allocate(nlayer_nbin);
  scatter_cross_section_inter.allocate(ninterface_nbin);
  planckband_lay.allocate(nlayer_plus2_nbin);
  planckband_int.allocate(ninterface_nbin);
  
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
}

// return device pointers for helios data save
// TODO: how ugly can it get, really?
void alfrodull_engine::get_device_pointers_for_helios_write(double *& dev_scat_cross_section_lay,
							    double *& dev_scat_cross_section_int,
							    double *& dev_interwave,
							    double *& dev_deltawave,
							    double *& dev_planck_lay,
							    double *& dev_planck_int,
							    double *& dev_planck_grid,
							    int & dim,
							    int & step
							    )
{
  dev_scat_cross_section_lay = *scatter_cross_section_lay;
  dev_scat_cross_section_int = *scatter_cross_section_inter;
  dev_interwave = *opacities.dev_opac_interwave;
  dev_deltawave = *opacities.dev_opac_deltawave;
  dev_planck_lay = *planckband_lay;
  dev_planck_int = *planckband_int;
  dev_planck_grid = *plancktable.planck_grid;
  dim = plancktable.dim;
  step = plancktable.step;
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
