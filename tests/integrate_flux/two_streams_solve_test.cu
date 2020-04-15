#include "cuda_device_memory.h"
#include "gauss_legendre_weights.h"
#include "integrate_flux.h"

#include "vector_operations.h"

#include <algorithm> // std::max
#include <cstdio>
#include <memory>
#include <random>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

using std::max;
using std::string;


void cuda_check_status_or_exit(const char* filename, const int& line) {
    cudaError_t err = cudaGetLastError();

    // Check device query
    if (err != cudaSuccess) {
        printf("[%s:%d] CUDA error check reports error: %s\n",
                    filename,
                    line,
                    cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

bool cmp_dbl(double d1, double d2, double eps) {
  if (d1 == d2)
    return true;
  else if (d1 == 0.0 && d2 == 0.0)
    return false;

  return fabs(d1 - d2) / fabs(max(d1, d2)) < eps;
}



// 2x2 matrix operations on linear stored by row matrices
// matrix is stored as
// A[0]  A[1]
// A[2]  A[3]

// __device__ void mult2x2(double * A, double * B, double * C) {
//   C[0] = A[0]*B[0] + A[1]*B[2];
//   C[1] = A[0]*B[1] + A[1]*B[3];
//   C[2] = A[2]*B[0] + A[3]*B[2];
//   C[3] = A[2]*B[1] + A[3]*B[3];
// }

// __device__ void add2x2(double * A, double * B, double * C) {
//   for (int i = 0; i < 4; i++)
//     C[i] = A[i] + B[i];
// }

// __device__ void sub2x2(double * A, double * B, double * C) {
//   for (int i = 0; i < 4; i++)
//     C[i] = A[i] - B[i];
// }

// // A - B*C
// __device__ multsub2x2(double * A, double * B, double * C, double * D) {
//   D[0] = A[0] - (B[0]*C[0] + B[1]*C[2]);
//   D[1] = A[1] - (B[0]*C[1] + B[1]*C[3]);
//   D[2] = A[2] - (B[2]*C[0] + B[3]*C[2]);
//   D[3] = A[3] - (B[2]*C[1] + B[3]*C[3]);
// }


// __device__ void invert2x2(double * A, double * A_inv) {
//   double det_inv = 1.0/( A[0]*A[3] - A[1]*A[2] );
//   A_inv[0] =  det_inv*A[3];
//   A_inv[1] = -det_inv*A[1];
//   A_inv[2] = -det_inv*A[2];
//   A_inv[3] =  det_inv*A[0];
// }

// matrix is stored as
// A.x  A.y
// A.z  A.w

// A - B*C

bool compare_vector2(double2 in, double2 ref, double epsilon, string message) {
  if ( !(cmp_dbl(in.x, ref.x, epsilon)
	 && cmp_dbl(in.y, ref.y, epsilon)))
    {
      printf(message.c_str());
      return false;
    }
  else
    {
      return true;
    }
}

bool compare_matrix4(double4 in, double4 ref, double epsilon, string message) {
  if ( !(cmp_dbl(in.x, ref.x, epsilon)
	 && cmp_dbl(in.y, ref.y, epsilon)
	 && cmp_dbl(in.z, ref.z, epsilon)
	 && cmp_dbl(in.w, ref.w, epsilon)))
    {
      printf(message.c_str());
      return false;
    }
  else
    {
      return true;
    }
}

bool test_vectors_ops(){
  printf("Running vector and matrix ops test\n");
  bool success = true;
  
  double epsilon = 1e-11;
  double4 M1 = make_double4(1.5, 0.5, 0.25, 2.0);
  double4 M2 = make_double4(3.0, 1.7, 0.7, 2.0);
  double2 V1 = make_double2(5.0, 2.7);
  double2 V2 = make_double2(6.0, 7.4);

  double4 negM1 = -M1;
  success &= compare_matrix4(negM1, make_double4(-1.5, -0.5, -0.25, -2.0), epsilon, "error in matrix negation\n");

  double4 invM1 = M1*inv2x2(M1);
  success &= compare_matrix4(invM1,
			     make_double4(1.0, 0.0, 0.0, 1.0),
			     epsilon,
			     "error in matrix inversion\n");
  
  double4 MulM1M2 = M1*M2;
  success &= compare_matrix4(MulM1M2,
			     make_double4(4.85, 3.55, 2.15, 4.425),
			     epsilon,
			     "error in matrix multiplication\n");

  double4 addM1M2 = M1 + M2;
  success &= compare_matrix4(addM1M2,
			     make_double4(4.5, 2.2, 0.95, 4.0),
			     epsilon,
			     "error in matrix addition\n");
  
  double4 subM1M2 = M1 - M2;
  success &= compare_matrix4(subM1M2,
			     make_double4(-1.5, -1.2, -0.45, 0.0),
			     epsilon,
			     "error in matrix subtraction\n");

  
  double2 MulM1V1 = M1*V1;
  success &= compare_vector2(MulM1V1,
			     make_double2(8.85, 6.65),
			     epsilon,
			     "error in matrix-vector multiplication\n");

  double2 addV1V2 = V1 + V2;
  success &= compare_vector2(addV1V2,
			     make_double2(11.0, 10.1),
			     epsilon,
			     "error in vector addition\n");
  
  double2 subV1V2 = V1 - V2;
  success &= compare_vector2(subV1V2,
			     make_double2(-1.0, -4.7),
			     epsilon,
			     "error in vector subtraction\n");

  double2 negV1 = -V1;
  success &= compare_vector2(negV1,
			     make_double2(-5.0, -2.7),
			     epsilon,
			     "error in vector negation\n");

  
  printf("Vector and matrix ops test done\n");

  return success;
}

__host__ __device__ void dbg_print_matrix(char * msg, double4 m)
{
  printf(msg, m.x, m.y, m.z, m.w);
}

__host__ __device__ void dbg_print_vector(char * msg, double2 v)
{
  printf(msg, v.x, v.y);
}

// Thomas solver for 2x2 matrix blocks
// N here is the number of matrices
__global__ void thomas_solve(double4 * A,
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

  C_prime[0] =  invB0 * C[0] ;
  D_prime[0] =  invB0 * D[0] ;
 // forward compute coefficients for matrix and RHS vector 
  for (int i = 1; i < N; i++)
    {
      double4 invBmACp = inv2x2(B[i] - (A[i]*C_prime[i-1]));

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



bool thomas_test() {
  printf("Running Thomas algorithm test\n");
  bool success = true;

  // number of rows
  int N = 120;
  // number of diagonals below main
  int d_b = 2;
  // number of diagonals above main
  int d_a = 2;
  // number of diagonals
  int diags = d_b + d_a + 1;
  // diagonal matrix
  std::shared_ptr<double[]> diag_M = std::shared_ptr<double[]>(new double[N*diags]);
  // solution vector (the one to find)
  std::shared_ptr<double[]> x_sol = std::shared_ptr<double[]>(new double[N]);
  // right hand side: diag_M*x_sol = d
  std::shared_ptr<double[]> d = std::shared_ptr<double[]>(new double[N]);

  
  // *********************************************************************************
  // Random number generator
  std::random_device rd;        // Will be used to obtain a seed for the random number engine
  std::mt19937       gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis_off_diag(0.0, 1.0);
  std::uniform_real_distribution<> dis_diag(5.0, 10.0);
  std::uniform_real_distribution<> dis_sol(1.0, 5.0);
  //  std::normal_distribution<> rand_norm_dist{1.0,0.5};
  // std::normal_distribution<> rand_norm_dist_plk{1e-5,0.5};    // Random number generator

  // *********************************************************************************
  // fill in matrix
  // should be diagonally dominant (sum of off-diagonal values should be smaller than diagonal)
  for (int r = 0; r < N; r++) {
    int d = 0;
    for (; d < d_b; d++)
      diag_M[r*diags + d] = dis_off_diag(gen);

    diag_M[r*diags + d] = dis_diag(gen);
    d++;
    
    for (; d < d_b + 1 + d_a; d++)
      diag_M[r*diags + d] = dis_off_diag(gen);
  }

  // corners
  diag_M[0*diags + 0] = 0.0;
  diag_M[0*diags + 1] = 0.0;
  diag_M[1*diags + 0] = 0.0;

  diag_M[(N-2)*diags + d_b + d_a]     = 0.0;
  diag_M[(N-1)*diags + d_b + d_a - 1] = 0.0;
  diag_M[(N-1)*diags + d_b + d_a]     = 0.0;
  
  // fill in solution
  for (int r = 0; r < N; r++) {
    x_sol[r] = dis_sol(gen);
  }
  
  // compute RHS
  // matrix multiplication
  for (int r = 0; r < N; r++) {    
    d[r] = 0.0;
    for (int c = -d_b; c < d_a + 1; c++) {
      if (r+c >= 0 && r+c < N )
	d[r] += diag_M[r*diags + d_b + c ]*x_sol[r+c];
    }
  }

  // *********************************************************************************
  // debug printout
  printf("d = np.array([\n");
  for (int r = 0; r < N; r++) {
    printf("%g,\n", d[r]);
  }
  printf("])\n");

  printf("x_sol = np.array([\n");
  for (int r = 0; r < N; r++) {
    printf("%g,\n", x_sol[r]);
  }
  printf("])\n");
    
  printf("A = np.array([\n");
  for (int r = 0; r < N; r++) {
    printf("[\t");
    for (int c = 0; c < N; c++) {
      if (r - c <= d_b && r - c  > 0) {
	printf("% 10g,\t", diag_M[r*diags + c - r + d_b]);
      }
      else if (c - r <= d_a && c - r > 0) {
	printf("% 10g,\t", diag_M[r*diags + c - r + d_b]);
      }
      else if (c - r == 0) {
	printf("% 10g,\t", diag_M[r*diags + c - r + d_b]);
      }
      else
	printf("% 10g,\t", 0.0);
    }
    printf("],\n");
  }
  printf("])\n");

    for (int r = 0; r < N; r++) {
    printf("[\t");
    for (int c = 0; c < diags; c++) {
      printf("% 10g\t", diag_M[r*diags + c]);
    }
    printf("],\n");
  }

    printf("\n");
  
    // debug printout
  printf("[\n");
  for (int r = 0; r < N; r++) {
    printf("[\t");
    for (int c = 0; c < N; c++) {
      if (r - c <= d_b && r - c  > 0) {
	printf("(% 3d,% 3d)\t", r, c - r );
      }
      else if (c - r <= d_a && c - r > 0) {
	printf("(% 3d,% 3d)\t", r, c - r );
      }
      else if (c - r == 0) {
	printf("(% 3d,% 3d)\t", r, c - r);
      }
      else
	printf("(% 3d,% 3d)\t", 0, 0);
    }
    printf("],\n");
  }
  printf("]\n");
  // *********************************************************************************
  // prepare GPU arrays
  cuda_device_memory<double> A;
  cuda_device_memory<double> B;
  cuda_device_memory<double> C;
  cuda_device_memory<double> D;
  cuda_device_memory<double> X;
  cuda_device_memory<double> C_prime;
  cuda_device_memory<double> D_prime;

  // be dumb and write out all dimensions
  // Number of 2x2 blocks in our block matrix
  int N_m = N/2;
  A.allocate((2*2)*N_m);
  B.allocate((2*2)*N_m);
  C.allocate((2*2)*N_m);

  C_prime.allocate((2*2)*N_m);
  D_prime.allocate((2*2)*N_m);
  
  X.allocate(2*N_m);
  D.allocate(2*N_m);

  const int R = 4;
  // fill in gpu data
  std::shared_ptr<double[]> A_h = A.get_host_data();
  // these are dummy values, shouldn't be used
  A_h[0] = 0.0;
  A_h[1] = 0.0;
  A_h[2] = 0.0;
  A_h[3] = 0.0;
  
  for (int i = 1; i < N_m; i++)
    {
      A_h[i*R + 0] = diag_M[2*i*diags + 0];
      A_h[i*R + 1] = diag_M[2*i*diags + 1];
      A_h[i*R + 2] = 0.0;
      A_h[i*R + 3] = diag_M[(2*i+1)*diags + 0];
      printf("A[%d]: [[ %g %g ], [ %g %g ]]\n", i,
	     A_h[i*R+0],
	     A_h[i*R+1],
	     A_h[i*R+2],
	     A_h[i*R+3]    );
    }
  A.put();
  
  std::shared_ptr<double[]> B_h = B.get_host_data();
  for (int i = 0; i < N_m; i++)
    {
      B_h[i*R + 0] = diag_M[2*i*diags + 2];
      B_h[i*R + 1] = diag_M[2*i*diags + 3];
      B_h[i*R + 2] = diag_M[(2*i+1)*diags + 1];
      B_h[i*R + 3] = diag_M[(2*i+1)*diags + 2];
    }
  B.put();
  
  std::shared_ptr<double[]> C_h = C.get_host_data();
  C_h[N_m - 1 + 0] = 0.0;
  C_h[N_m - 1 + 1] = 0.0;
  C_h[N_m - 1 + 2] = 0.0;
  C_h[N_m - 1 + 3] = 0.0;
  for (int i = 0; i < N_m; i++)
    {
      C_h[i*R + 0] = diag_M[2*i*diags + 4];
      C_h[i*R + 1] = 0.0;
      C_h[i*R + 2] = diag_M[(2*i+1)*diags + 3];
      C_h[i*R + 3] = diag_M[(2*i+1)*diags + 4];
    }
  C.put();

  std::shared_ptr<double[]> D_h = D.get_host_data();
  for (int i = 0; i < N_m; i++)
    {
      D_h[i*2 + 0] = d[2*i + 0];
      D_h[i*2 + 1] = d[2*i + 1];
    }
  D.put();
  

  
  
  // run thomas algorithm
  dim3 block(1, 1, 1);
  dim3 grid(1, 1, 1);
  thomas_solve<<<grid, block>>>((double4*)*A,
				(double4*)*B,
				(double4*)*C,
				(double2*)*D,
				(double4*)*C_prime,
				(double2*)*D_prime,
				(double2*)*X, N_m);
  
  
  
  // *********************************************************************************
  // check solution
  bool debug = true;
  int error = 0;
  int total = 0;
  
  std::shared_ptr<double[]> X_h   = X.get_host_data();
  
  double epsilon = 1e-12;
  for (int i = 0; i < N; i++) {
    double x     = X_h[i];
    double x_ref = x_sol[i];
    
    bool match = cmp_dbl(x, x_ref, epsilon);
    
    
    if (match) {
      if (debug)
	printf("% 5d % 20.12g == % 20.12g %d\n",
	       i, x, x_ref, match);
    }
    else {
      error += 1;
      printf("% 5d % 20.12g == % 20.12g %d\n",
	     i, x, x_ref, match);
    }
    total += 1;
    success &= match;
  }
  
  printf("errors: %d/%d\n", error, total);
  
  printf("Finished thomas algorithm test\n");
  return success;
}


bool two_streams_solver_test()
{
  bool success = true;
  printf("Running Two Streams Solver test\n");
  
  
  int point_num = 1;

  int nlayer = 15;

    int nbin = 1;
    int ny   = 1;

    int ninterface         = nlayer + 1;
    int nlayer_nbin        = nlayer * nbin;
    int nlayer_plus2_nbin  = (nlayer + 2) * nbin;
    int nlayer_wg_nbin     = nlayer * nbin * ny;
    int ninterface_nbin    = ninterface * nbin;
    int ninterface_wg_nbin = ninterface * ny * nbin;

    printf("nlayer: %d\n", nlayer);
    printf("ninterface: %d\n", ninterface);
    printf("nbin: %d\n", nbin);
    printf("ny: %d\n", ny);
    // initialise arrays

    bool iso = true;
	
    cuda_device_memory<double> F_down_wg_helios;
    cuda_device_memory<double> F_up_wg_helios;

    cuda_device_memory<double> F_down_wg_alf;
    cuda_device_memory<double> F_up_wg_alf;

    cuda_device_memory<double> F_dir_wg;

    cuda_device_memory<double> planckband_lay;
    cuda_device_memory<double> w_0;
    cuda_device_memory<double> M_term;
    cuda_device_memory<double> N_term;
    cuda_device_memory<double> P_term;
    cuda_device_memory<double> G_plus;
    cuda_device_memory<double> G_minus;
    cuda_device_memory<double> g_0_tot_lay;

    double g_0 = 0.5;
    bool single_walk = true;
    double Rstar = 1.0;
    double a = 1.0;
    double f_factor = 1.0;
    double mu_star = -1.0;
    double epsi = 0.5;
    bool dir_beam = false;
    bool clouds = false;
    bool scat_corr = true;
    double albedo = 1.0;
    double i2s_transition = 0.1;

    bool debug = true;

    F_down_wg_helios.allocate(point_num * ninterface_wg_nbin);
    F_up_wg_helios.allocate(point_num * ninterface_wg_nbin);

    F_down_wg_alf.allocate(point_num * ninterface_wg_nbin);
    F_up_wg_alf.allocate(point_num * ninterface_wg_nbin);

    F_dir_wg.allocate(point_num * ninterface_wg_nbin);

    F_down_wg_helios.zero();
    F_up_wg_helios.zero();

    F_down_wg_alf.zero();
    F_up_wg_alf.zero();
    
    
    planckband_lay.allocate(nlayer_plus2_nbin);

    // Random number generator
    std::random_device rd;        //Will be used to obtain a seed for the random number engine
    std::mt19937       gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> rand_norm_dist{1.0,0.5};

    std::normal_distribution<> rand_norm_dist_plk{1e-5,0.5};
	
    if (iso) {
      printf("Initialising data for ISO case\n");
        M_term.allocate(nlayer_wg_nbin);
        N_term.allocate(nlayer_wg_nbin);
        P_term.allocate(nlayer_wg_nbin);
        G_plus.allocate(nlayer_wg_nbin);
        G_minus.allocate(nlayer_wg_nbin);
        w_0.allocate(nlayer_wg_nbin);
	g_0_tot_lay.allocate(nlayer_nbin);

	// initialise data
	// get data pointers
	std::shared_ptr<double[]> M_term_h      = M_term.get_host_data_ptr();
	std::shared_ptr<double[]> N_term_h      = N_term.get_host_data_ptr();
	std::shared_ptr<double[]> P_term_h      = P_term.get_host_data_ptr();
	std::shared_ptr<double[]> G_plus_h      = G_plus.get_host_data_ptr();
	std::shared_ptr<double[]> G_minus_h     = G_minus.get_host_data_ptr();
	std::shared_ptr<double[]> w_0_h         = w_0.get_host_data_ptr();
	std::shared_ptr<double[]> g_0_tot_lay_h = g_0_tot_lay.get_host_data_ptr();
	std::shared_ptr<double[]> planckband_lay_h = planckband_lay.get_host_data_ptr();
	       
	for (int i = 0; i < nlayer; i++) {
	  for (int j = 0; j < nbin; j++) {
	    for (int k = 0; k < ny; k++) {
	      // printf("%d %d %d \n", i, j, k);
	      double T = fabs(rand_norm_dist(gen));
	      double z_p = fabs(rand_norm_dist(gen));
	      double z_m = fabs(rand_norm_dist(gen));

	      M_term_h[k + ny*j + i*ny*nbin] = z_m*z_m*T*T - z_p*z_p;

	      double n_term = 0.0;
	      if (scat_corr)
		n_term = z_m*z_p*(1.0 - T*T);
	      
	      N_term_h[k + ny*j + i*ny*nbin] = n_term;
	      
	      P_term_h[k + ny*j + i*ny*nbin] = (z_m*z_m - z_p*z_p)*T;
	      
	      double a = fabs(rand_norm_dist(gen));
	      double b = fabs(rand_norm_dist(gen));
	      G_plus_h[k + ny*j + i*ny*nbin] = max(a, b);
	      G_minus_h[k + ny*j + i*ny*nbin] = min(a,b);
	      w_0_h[k + ny*j + i*ny*nbin] = 0.5; //dis(gen);
	      // printf("%g %g %g %g %g %g %g\n",
	      // 	     M_term_h[k + ny*j + i*ny*nbin],
	      // 	     N_term_h[k + ny*j + i*ny*nbin],
	      // 	     P_term_h[k + ny*j + i*ny*nbin],
	      // 	     G_plus_h[k + ny*j + i*ny*nbin],
	      // 	     G_minus_h[k + ny*j + i*ny*nbin],
	      // 	     w_0_h[k + ny*j + i*ny*nbin]);
	    }
	    g_0_tot_lay_h[j + i*nbin] = rand_norm_dist(gen);
	  }
	}

	for (int i = 0; i < nlayer; i++) {
	  for (int j = 0; j < nbin; j++) {
	    planckband_lay_h[j*(ninterface -1+2) + i] = 0.0;
	    //planckband_lay_h[j*(ninterface -1+2) + i] = fabs(rand_norm_dist(gen));
	  }
	}
	for (int j = 0; j < nbin; j++) {
	  planckband_lay_h[j*(ninterface -1+2) + nlayer ] = fabs(rand_norm_dist_plk(gen));
	  planckband_lay_h[j*(ninterface -1+2) + nlayer + 1] = fabs(rand_norm_dist_plk(gen));
	}
	
	M_term.put();
	N_term.put();
	P_term.put();
	G_plus.put();
	G_minus.put();
	w_0.put();
	g_0_tot_lay.put();
	planckband_lay.put();

	
    }
    else {
      // M_upper.allocate(nlayer_wg_nbin);
      // M_lower.allocate(nlayer_wg_nbin);
      // N_upper.allocate(nlayer_wg_nbin);
        // N_lower.allocate(nlayer_wg_nbin);
        // P_upper.allocate(nlayer_wg_nbin);
        // P_lower.allocate(nlayer_wg_nbin);
        // G_plus_upper.allocate(nlayer_wg_nbin);
        // G_plus_lower.allocate(nlayer_wg_nbin);
        // G_minus_upper.allocate(nlayer_wg_nbin);
        // G_minus_lower.allocate(nlayer_wg_nbin);
        // w_0_upper.allocate(nlayer_wg_nbin);
        // w_0_lower.allocate(nlayer_wg_nbin);
    }    

    {
      std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
          
      // compute
      printf("Original version\n");
      
      bool debug_fn = false;
      
      int nscat_step = 0;
      if (single_walk)
        nscat_step = 200;
      else
        nscat_step = 3;
      
      for (int scat_iter = 0; scat_iter < nscat_step * scat_corr + 1; scat_iter++) {
        if (iso) {
	  printf("Loop\n");
	  dim3 block(16, 16, 1);
	  dim3 grid((nbin + 15) / 16, (ny + 15) / 16, 1);
	  fband_iso_notabu<<<grid, block>>>(*F_down_wg_helios,
					    *F_up_wg_helios,
					    *F_dir_wg,
					    *planckband_lay,
					    *w_0,
					    *M_term,
					    *N_term,
					    *P_term,
					    *G_plus,
					    *G_minus,
					    *g_0_tot_lay,
					    g_0,
					    single_walk,
					    Rstar,
					    a,
					    ninterface,
					    nbin,
					    f_factor,
					    mu_star,
					    ny,
					    epsi,
					    dir_beam,
					    clouds,
					    scat_corr,
					    albedo,
					    debug_fn,
					    i2s_transition);
	  
	  cudaDeviceSynchronize();
	  cuda_check_status_or_exit(__FILE__, __LINE__);
        }
        else {
	  /*
	    int nbin = opacities.nbin;
	    int ny   = opacities.ny;
	    
	    dim3 block(16, 16, 1);
	    
	    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, 1);
	    
	  // calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
	  fband_noniso_notabu<<<grid, block>>>(F_down_wg,
	  F_up_wg,
					       Fc_down_wg,
					       Fc_up_wg,
					       F_dir_wg,
					       Fc_dir_wg,
					       *planckband_lay,
					       *planckband_int,
					       *w_0_upper,
					       *w_0_lower,
					       *delta_tau_wg_upper,
					       *delta_tau_wg_lower,
					       *M_upper,
					       *M_lower,
					       *N_upper,
					       *N_lower,
					       *P_upper,
					       *P_lower,
					       *G_plus_upper,
					       *G_plus_lower,
					       *G_minus_upper,
					       *G_minus_lower,
					       g_0_tot_lay,
					       g_0_tot_int,
					       g_0,
					       singlewalk,
					       Rstar,
					       a,
					       ninterface,
					       nbin,
					       f_factor,
					       mu_star,
					       ny,
					       epsi,
					       delta_tau_limit,
					       dir_beam,
					       clouds,
					       scat_corr,
					       albedo,
					       debug,
					       i2s_transition);
	  */
        }
      }

      std::chrono::system_clock::time_point stop  = std::chrono::system_clock::now();
      auto duration_helios = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

      printf("Computed in: HELIOS: %ld us\n", duration_helios.count());
      
    }
    cuda_check_status_or_exit(__FILE__, __LINE__);

    
    printf("Comparing output\n");
    

    int error = 0;
    int total = 0;

    std::shared_ptr<double[]> F_up_wg_helios_h   = F_up_wg_helios.get_host_data();
    std::shared_ptr<double[]> F_down_wg_helios_h   = F_down_wg_helios.get_host_data();
    std::shared_ptr<double[]> F_up_wg_alf_h   = F_up_wg_alf.get_host_data();
    std::shared_ptr<double[]> F_down_wg_alf_h   = F_down_wg_alf.get_host_data();
    
    double epsilon = 1e-12;
    for (int i = 0; i < ninterface; i++) {
      for (int j = 0; j < nbin; j++) {
	for (int k = 0; k < ny; k++) {
	  double f_dwn_wg_helios     = F_down_wg_helios_h[i * nbin * ny + j * ny + k];
	  double f_up_wg_helios     = F_up_wg_helios_h[i * nbin * ny + j * ny + k];

	  double f_dwn_wg_alf     = F_down_wg_alf_h[i * nbin * ny + j * ny + k];
	  double f_up_wg_alf     = F_up_wg_alf_h[i * nbin * ny + j * ny + k];
	  

	  bool match_dwn = cmp_dbl(f_dwn_wg_helios, f_dwn_wg_alf, epsilon);
	  bool match_up = cmp_dbl(f_up_wg_helios, f_up_wg_alf, epsilon);
	  
	  if (match_up && match_dwn) {
	    if (debug)
	      printf("% 5d % 5d % 5d % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d\n",
		     i,j,k,
		     f_dwn_wg_helios,
		     f_dwn_wg_alf,
		     match_dwn,
		     f_up_wg_helios,
		     f_up_wg_alf,
		     match_up);
            }
            else {
                error += 1;
		printf("% 5d % 5d % 5d % 20.12g == % 20.12g, %5d - % 20.12g == % 20.12g, %5d\n",
		       i,j,k,
		       f_dwn_wg_helios,
		       f_dwn_wg_alf,
		       match_dwn,
		       f_up_wg_helios,
		       f_up_wg_alf,
		       match_up);
            }
            total += 1;
	    success &= match_up && match_dwn;
        }
      }
    }
    printf("errors: %d/%d\n", error, total);
    
    printf("Two stream solver test done\n");
    return success;
}

int main(int argc, char** argv) {

  bool success = true;
  success &= test_vectors_ops();
  success &= thomas_test();
  // success &= two_streams_solver_test();
  

    if (success) {
        printf("Success\n");
        return 0;
    }
    else {
        printf("Fail\n");
        return -1;
    }
}
