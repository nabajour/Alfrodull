#include "math_helpers.h"


// calculates analytically the integral of the planck function
__device__ double analyt_planck(
        int 	n, 
        double	y1, 
        double	y2)
{

  double dn=n;

  return exp(-dn*y2) * ((y2*y2*y2)/dn + 3.0*(y2*y2)/(dn*dn) + 6.0*y2/(dn*dn*dn) + 6.0/(dn*dn*dn*dn))
    - exp(-dn*y1) * ((y1*y1*y1)/dn + 3.0*(y1*y1)/(dn*dn) + 6.0*y1/(dn*dn*dn) + 6.0/(dn*dn*dn*dn));
}
