#include "math_helpers.h"
#include "vector_operations.h"

// calculates analytically the integral of the planck function
__device__ double analyt_planck(int n, double y1, double y2) {

    double dn = n;

    return exp(-dn * y2)
               * ((y2 * y2 * y2) / dn + 3.0 * (y2 * y2) / (dn * dn) + 6.0 * y2 / (dn * dn * dn)
                  + 6.0 / (dn * dn * dn * dn))
           - exp(-dn * y1)
                 * ((y1 * y1 * y1) / dn + 3.0 * (y1 * y1) / (dn * dn) + 6.0 * y1 / (dn * dn * dn)
                    + 6.0 / (dn * dn * dn * dn));
}


// calculates the power operation with a foor loop -- is allegedly faster than the implemented pow() function
__device__ double power_int(double x, int i) {

    double result = 1.0;
    int    j      = 1;

    while (j <= i) {
        result *= x;
        j++;
    }
    return result;
}

// Thomas solver for 2x2 matrix blocks
// N here is the number of matrices
__host__ __device__ void thomas_solve(double4 * A,
			     double4 * B,
			     double4 * C,
			     double2 * D,
			     double4 * C_prime,
			     double2 * D_prime,
			     double2 * X,
			     int N)
{
  // initialise
  double4 invB0 = inv2x2(B[0]);

  // printf("invB0[ %d ]: [[ %g, %g ], [ %g, %g ]] \n",
  // 	 0, invB0.x, invB0.y, invB0.z, invB0.w);
	
  C_prime[0] =  invB0 * C[0] ;
  D_prime[0] =  invB0 * D[0] ;
 // forward compute coefficients for matrix and RHS vector 
  for (int i = 1; i < N; i++)
    {
      double4 BmACp = B[i] - (A[i]*C_prime[i-1]);
      // printf("B[ %d ]: [[ %g, %g ], [ %g, %g ]] \nA[ %d ]: [[ %g, %g ], [ %g, %g ]] \nCp[ %d ]: [[ %g, %g ], [ %g, %g ]] \nBmACp[ %d ]: [[ %g, %g ], [ %g, %g ]] \n",
      // 	     i,  
      // 	     B[i].x, B[i].y, B[i].z, B[i].w,
      // 	     i,A[i].x, A[i].y, A[i].z, A[i].w,
      // 	     i,C_prime[i-1].x, C_prime[i-1].y, C_prime[i-1].z, C_prime[i-1].w,
      // 	     i,BmACp.x, BmACp.y, BmACp.z, BmACp.w);
      double4 invBmACp = inv2x2(BmACp);
	//double4 invBmACp = inv2x2(B[i] - (A[i]*C_prime[i-1]));
      // if (isnan(invBmACp.x) || isnan(invBmACp.y) || isnan(invBmACp.z) || isnan(invBmACp.w))
      // 	printf("thomas_solve nan inv matrix at %d: [[ %g, %g ], [ %g, %g ]] \n",
      // 	       i,
      // 	       invBmACp.x,
      // 	       invBmACp.y,
      // 	       invBmACp.z,
      // 	       invBmACp.w
      // 	       );
      if (i < N - 1)
	{
	  C_prime[i] = invBmACp*C[i];
	}
      D_prime[i] = invBmACp*(D[i] - A[i]*D_prime[i-1]);
    }

  

  // Back substitution
  // last case:
  X[N-1] = D_prime[N-1];
  for (int i = N-2; i>= 0; i--) {
    X[i] = D_prime[i] - C_prime[i]*X[i+1];
  }
}


