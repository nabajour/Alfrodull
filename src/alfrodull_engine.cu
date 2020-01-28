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
