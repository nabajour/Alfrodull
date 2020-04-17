__host__ double alf_sign(double val);

__host__ double alf_solve_kepler(double mean_anomaly, double ecc);

__host__ double alf_calc_r_orb(double ecc_anomaly, double ecc);

__host__ double alf_ecc2true_anomaly(double ecc_anomaly, double ecc);

__host__ double alf_true2ecc_anomaly(double true_anomaly, double ecc);

__device__ double
alf_calc_zenith(double *     lonlat_d, //latitude/longitude grid
            const double alpha,    //current RA of star (relative to zero long on planet)
            const double alpha_i,
            const double sin_decl, //declination of star
            const double cos_decl,
            const bool   sync_rot,
            const double ecc,
            const double obliquity,
            const int    id);

__global__ void alf_annual_insol(double *insol_ann_d, double *insol_d, int nstep, int num);
