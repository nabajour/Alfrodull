

__host__ double alf_sign(double val) {
    if (val < 0.0)
        return -1.0;
    else
        return 1.0;
}

__host__ double alf_solve_kepler(double mean_anomaly, double ecc) {

    // Solve Kepler's equation (see Murray & Dermott 1999)
    // Get eccentric anomaly from mean anomaly and eccentricity

    double ecc_anomaly, fi, fi_1, fi_2, fi_3, di_1, di_2, di_3;

    ecc_anomaly = mean_anomaly + alf_sign(sin(mean_anomaly)) * 0.85 * ecc;
    di_3        = 1.0;

    while (di_3 > 1e-15) {
        fi          = ecc_anomaly - ecc * sin(ecc_anomaly) - mean_anomaly;
        fi_1        = 1.0 - ecc * cos(ecc_anomaly);
        fi_2        = ecc * sin(ecc_anomaly);
        fi_3        = ecc * cos(ecc_anomaly);
        di_1        = -fi / fi_1;
        di_2        = -fi / (fi_1 + 0.5 * di_1 * fi_2);
        di_3        = -fi / (fi_1 + 0.5 * di_2 * fi_2 + 1. / 6. * di_2 * di_2 * fi_3);
        ecc_anomaly = ecc_anomaly + di_3;
    }
    return ecc_anomaly;
}

__host__ double alf_calc_r_orb(double ecc_anomaly, double ecc) {

    // Calc relative distance between planet and star (units of semi-major axis)

    double r = 1.0 - ecc * cos(ecc_anomaly);
    return r;
}

__host__ double alf_ecc2true_anomaly(double ecc_anomaly, double ecc) {

    // Convert from eccentric to true anomaly

    double tanf2, true_anomaly;
    tanf2        = sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(ecc_anomaly / 2.0);
    true_anomaly = 2.0 * atan(tanf2);
    if (true_anomaly < 0.0)
        true_anomaly += 2 * M_PI;
    return true_anomaly;
}

__host__ double alf_true2ecc_anomaly(double true_anomaly, double ecc) {

    // Convert from true to eccentric anomaly

    double cosE, ecc_anomaly;
    while (true_anomaly < 0.0)
        true_anomaly += 2 * M_PI;
    while (true_anomaly >= 2 * M_PI)
        true_anomaly -= 2 * M_PI;
    cosE = (cos(true_anomaly) + ecc) / (1.0 + ecc * cos(true_anomaly));
    if (true_anomaly < M_PI) {
        ecc_anomaly = acos(cosE);
    }
    else {
        ecc_anomaly = 2 * M_PI - acos(cosE);
    }
    return ecc_anomaly;
}

__device__ double
alf_calc_zenith(double *     lonlat_d, //latitude/longitude grid
            const double alpha,    //current RA of star (relative to zero long on planet)
            const double alpha_i,
            const double sin_decl, //declination of star
            const double cos_decl,
            const bool   sync_rot,
            const double ecc,
            const double obliquity,
            const int    id) {

    // Calculate the insolation (scaling) at a point on the surface

    double coszrs;

    if (sync_rot) {
        if (ecc < 1e-10) {
            if (obliquity < 1e-10) { //standard sync, circular, zero obl case
                coszrs = cos(lonlat_d[id * 2 + 1]) * cos(lonlat_d[id * 2] - alpha_i);
            }
            else { //sync, circular, but some obliquity
                coszrs = (sin(lonlat_d[id * 2 + 1]) * sin_decl
                          + cos(lonlat_d[id * 2 + 1]) * cos_decl * cos(lonlat_d[id * 2] - alpha_i));
            }
        }
        else {                       //in below cases, watch out for numerical drift of mean(alpha)
            if (obliquity < 1e-10) { // sync, zero obliquity, but ecc orbit
                coszrs = cos(lonlat_d[id * 2 + 1]) * cos(lonlat_d[id * 2] - alpha);
            }
            else { // sync, non-zero obliquity, ecc orbit (full calculation applies)
                coszrs = (sin(lonlat_d[id * 2 + 1]) * sin_decl
                          + cos(lonlat_d[id * 2 + 1]) * cos_decl * cos(lonlat_d[id * 2] - alpha));
            }
        }
    }
    else {
        coszrs = (sin(lonlat_d[id * 2 + 1]) * sin_decl
                  + cos(lonlat_d[id * 2 + 1]) * cos_decl * cos(lonlat_d[id * 2] - alpha));
    }
    return coszrs; //zenith angle
}

__global__ void alf_annual_insol(double *insol_ann_d, double *insol_d, int nstep, int num) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < num) {
        insol_ann_d[id] = insol_ann_d[id] * (nstep - 1) / nstep + insol_d[id] / nstep;
    }
}

