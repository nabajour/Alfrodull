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
				      const bool & iso_)
{
  nlayer = nlayer_;
  ninterface = nlayer + 1;
  iso = iso_;

  // TODO: maybe should stay in opacities object
  nbin = opacities.nbin;
}

void alfrodull_engine::allocate_internal_variables()
{
  // scatter cross section layer and interface

  scatter_cross_section_lay.allocate(nlayer*opacities.nbin);
  scatter_cross_section_inter.allocate(ninterface*opacities.nbin);
}

// return device pointers for helios data save
// TODO: how ugly can it get, really?
void alfrodull_engine::get_device_pointers_for_helios_write(double *& dev_scat_cross_section_lay,
							    double *& dev_scat_cross_section_int,
							    double *& dev_interwave,
							    double *& dev_deltawave)
{
  dev_scat_cross_section_lay = *scatter_cross_section_lay;
  dev_scat_cross_section_int = *scatter_cross_section_inter;
  dev_interwave = *opacities.dev_opac_interwave;
  dev_deltawave = *opacities.dev_opac_deltawave;
}
