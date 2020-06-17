#include "math_helpers.h"
#include "vector_operations.h"

#include "cuda_device_memory.h"

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
__host__ __device__ void thomas_solve(double4* A,
                                      double4* B,
                                      double4* C,
                                      double2* D,
                                      double4* C_prime,
                                      double2* D_prime,
                                      double2* X,
                                      int      N) {
    // initialise
    double4 invB0 = inv2x2(B[0]);

    // printf("invB0[ %d ]: [[ %g, %g ], [ %g, %g ]] \n",
    // 	 0, invB0.x, invB0.y, invB0.z, invB0.w);

    C_prime[0] = invB0 * C[0];
    D_prime[0] = invB0 * D[0];
    // forward compute coefficients for matrix and RHS vector
    for (int i = 1; i < N; i++) {
        double4 BmACp = B[i] - (A[i] * C_prime[i - 1]);
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
        if (i < N - 1) {
            C_prime[i] = invBmACp * C[i];
        }
        D_prime[i] = invBmACp * (D[i] - A[i] * D_prime[i - 1]);
    }


    // Back substitution
    // last case:
    X[N - 1] = D_prime[N - 1];
    for (int i = N - 2; i >= 0; i--) {
        X[i] = D_prime[i] - C_prime[i] * X[i + 1];
    }
}


// first simple integration over weights
__global__ void integrate_val_band(double* val_wg,       // in
                                   double* val_band,     // out
                                   double* gauss_weight, // in
                                   int     nbin,
                                   int     num_val,
                                   int     ny) {

    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;


    if (val_idx < num_val && bin_idx < nbin) {
        // set memory to 0.

        val_band[bin_idx + nbin * val_idx] = 0;


        int bin_offset = bin_idx + nbin * val_idx;

        for (int y = 0; y < ny; y++) {
            double w             = gauss_weight[y];
            int    weight_offset = y + ny * bin_idx + ny * nbin * val_idx;

            val_band[bin_offset] += 0.5 * w * val_wg[weight_offset];
        }
    }
}

// simple integration over bins/bands
__global__ void integrate_val_tot(double* val_tot,     // out
                                  double* val_band,    // in
                                  double* deltalambda, // in
                                  int     nbin,
                                  int     num_val) {


    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (val_idx < num_val) {

        val_tot[val_idx] = 0.0;

        for (int bin = 0; bin < nbin; bin++) {
            int band_idx = val_idx * nbin + bin;
            val_tot[val_idx] += val_band[band_idx] * deltalambda[bin];
        }
    }
}

std::shared_ptr<double[]>
integrate_band(double* val, double* gauss_weight, int num_val, int nbin, int ny) {
    cuda_device_memory<double> val_band;

    val_band.allocate(num_val * nbin);

    {
        int  num_levels_per_block = 256 / nbin + 1;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block, nbin);

        integrate_val_band<<<gridsize, blocksize>>>(val,          // in
                                                    *val_band,    // out
                                                    gauss_weight, // in
                                                    nbin,
                                                    num_val,
                                                    ny);

        cudaDeviceSynchronize();
    }
    return val_band.get_host_data();
}

std::shared_ptr<double[]> integrate_wg_band(double* val,
                                            double* gauss_weight,
                                            double* deltalambda,
                                            int     num_val,
                                            int     nbin,
                                            int     ny) {
    cuda_device_memory<double> val_band;
    cuda_device_memory<double> val_tot;

    val_band.allocate(num_val * nbin);

    val_tot.allocate(num_val);

    {
        int  num_levels_per_block = 256 / nbin + 1;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block, nbin);

        integrate_val_band<<<gridsize, blocksize>>>(val,          // in
                                                    *val_band,    // out
                                                    gauss_weight, // in
                                                    nbin,
                                                    num_val,
                                                    ny);

        cudaDeviceSynchronize();
    }

    {
        int  num_levels_per_block = 256;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block);
        integrate_val_tot<<<gridsize, blocksize>>>(*val_tot,    // out
                                                   *val_band,   // in
                                                   deltalambda, // in
                                                   nbin,
                                                   num_val);

        cudaDeviceSynchronize();
    }

    return val_tot.get_host_data();
}

__global__ void arrays_mean(double* array1, double* array2, double* array_out, int array_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < array_size) {
        array_out[idx] = (array1[idx] + array2[idx]) / 2.0;
    }
}
