#include "opacities.h"

class alfrodull_engine
{
public:
  alfrodull_engine();

  void init();
  
  void load_opacities(const string & filename);
  
private:
  opacity_table opacity;
  
};
