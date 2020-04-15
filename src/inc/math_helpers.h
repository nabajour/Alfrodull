#pragma once

__device__ double analyt_planck(int n, double y1, double y2);


__device__ double power_int(double x, int i);

__host__ __device__ void thomas_solve(double4 * A,
						 double4 * B,
						 double4 * C,
						 double2 * D,
						 double4 * C_prime,
						 double2 * D_prime,
						 double2 * X,
						 int N);
