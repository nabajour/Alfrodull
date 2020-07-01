#include "alfrodull_engine.h"
#include "gauss_legendre_weights.h"

#include "calculate_physics.h"
#include "inc/cloud_opacities.h"
#include "integrate_flux.h"
#include "interpolate_values.h"

#include "binary_test.h"
#include "debug.h"

#include <functional>
#include <map>

#include "math_helpers.h"

using std::string;


void cuda_check_status_or_exit(const char* filename, int line) {
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

alfrodull_engine::alfrodull_engine() {
    printf("Creating Alfrodull engine\n");
}

void alfrodull_engine::load_opacities(const string& filename) {
    printf("Loading opacities from %s\n", filename.c_str());

    opacities.load_opacity_table(filename);
}

void alfrodull_engine::init() {
    printf("Alfrodull Init\n");

    load_opacities("input/opac_sample.h5");
}

void alfrodull_engine::set_parameters(const int&    nlayer_,
                                      const bool&   iso_,
                                      const double& T_star_,
                                      const bool&   real_star_,
                                      const double& fake_opac_,
                                      const double& g_0_,
                                      const double& epsi_,
                                      const double& epsilon_2_,
                                      const bool&   scat_,
                                      const bool&   scat_corr_,
                                      const double& R_planet_,
                                      const double& R_star_,
                                      const double& a_,
                                      const bool&   dir_beam_,
                                      const bool&   geom_zenith_corr_,
                                      const double& f_factor_,
                                      const double& w_0_limit_,
                                      const double& i2s_transition_,
                                      const double& mu_star_limit_,
                                      const int&    max_num_parallel_columns_,
                                      const bool&   debug_) {
    nlayer     = nlayer_;
    ninterface = nlayer + 1;
    iso        = iso_;
    T_star     = T_star_;

    real_star        = real_star_;
    fake_opac        = fake_opac_;
    g_0              = g_0_;
    epsi             = epsi_;
    epsilon2         = epsilon_2_;
    scat             = scat_;
    scat_corr        = scat_corr_;
    R_planet         = R_planet_;
    R_star           = R_star_;
    a                = a_;
    dir_beam         = dir_beam_;
    geom_zenith_corr = geom_zenith_corr_;
    f_factor         = f_factor_;
    w_0_limit        = w_0_limit_;

    i2s_transition = i2s_transition;
    debug          = debug_;
    mu_star_limit  = mu_star_limit_;

    max_num_parallel_columns = max_num_parallel_columns_;
    // TODO: maybe should stay in opacities object
    //    nbin = opacities.nbin;

    // prepare_planck_table();
}

void alfrodull_engine::allocate_internal_variables() {
    int nlayer_nbin        = nlayer * opacities.nbin;
    int nlayer_plus2_nbin  = (nlayer + 2) * opacities.nbin;
    int ninterface_nbin    = ninterface * opacities.nbin;
    int nlayer_wg_nbin     = nlayer * opacities.ny * opacities.nbin;
    int ninterface_wg_nbin = ninterface * opacities.ny * opacities.nbin;
    int num_cols           = max_num_parallel_columns;
    if (iso) {
        delta_tau_wg.allocate(num_cols * nlayer_wg_nbin);
    }
    else {
        delta_tau_wg_upper.allocate(num_cols * nlayer_wg_nbin);
        delta_tau_wg_lower.allocate(num_cols * nlayer_wg_nbin);
    }

    if (thomas) {
        if (iso) {
            A_buff.allocate(num_cols * ninterface_wg_nbin * 4);       // thomas worker
            B_buff.allocate(num_cols * ninterface_wg_nbin * 4);       // thomas worker
            C_buff.allocate(num_cols * ninterface_wg_nbin * 4);       // thomas worker
            D_buff.allocate(num_cols * ninterface_wg_nbin * 2);       // thomas worker
            C_prime_buff.allocate(num_cols * ninterface_wg_nbin * 4); // thomas worker
            D_prime_buff.allocate(num_cols * ninterface_wg_nbin * 2); // thomas worker
            X_buff.allocate(num_cols * ninterface_wg_nbin * 2);       // thomas worker
        }
        else {
            int num_th_layers             = nlayer * 2;
            int num_th_interfaces         = num_th_layers + 1;
            int num_th_interfaces_wg_nbin = num_th_interfaces * opacities.ny * opacities.nbin;
            A_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 4);       // thomas worker
            B_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 4);       // thomas worker
            C_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 4);       // thomas worker
            D_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 2);       // thomas worker
            C_prime_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 4); // thomas worker
            D_prime_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 2); // thomas worker
            X_buff.allocate(num_cols * num_th_interfaces_wg_nbin * 2);       // thomas worker
        }
    }
    // flux computation internal quantities
    if (iso) {
        M_term.allocate(num_cols * nlayer_wg_nbin);
        N_term.allocate(num_cols * nlayer_wg_nbin);
        P_term.allocate(num_cols * nlayer_wg_nbin);
        G_plus.allocate(num_cols * nlayer_wg_nbin);
        G_minus.allocate(num_cols * nlayer_wg_nbin);
        w0_wg.allocate(num_cols * nlayer_wg_nbin);
        g0_wg.allocate(num_cols * nlayer_wg_nbin);

        g0_band.allocate(num_cols * nlayer_nbin);
        w0_band.allocate(num_cols * nlayer_nbin);

        delta_col_mass.allocate(num_cols * nlayer);
    }
    else {
        M_upper.allocate(num_cols * nlayer_wg_nbin);
        M_lower.allocate(num_cols * nlayer_wg_nbin);
        N_upper.allocate(num_cols * nlayer_wg_nbin);
        N_lower.allocate(num_cols * nlayer_wg_nbin);
        P_upper.allocate(num_cols * nlayer_wg_nbin);
        P_lower.allocate(num_cols * nlayer_wg_nbin);
        G_plus_upper.allocate(num_cols * nlayer_wg_nbin);
        G_plus_lower.allocate(num_cols * nlayer_wg_nbin);
        G_minus_upper.allocate(num_cols * nlayer_wg_nbin);
        G_minus_lower.allocate(num_cols * nlayer_wg_nbin);

        // for computations
        g0_wg_upper.allocate(num_cols * nlayer_wg_nbin);
        g0_wg_lower.allocate(num_cols * nlayer_wg_nbin);
        w0_wg_upper.allocate(num_cols * nlayer_wg_nbin);
        w0_wg_lower.allocate(num_cols * nlayer_wg_nbin);

        // used to store layer value for output
        w0_wg.allocate(num_cols * nlayer_wg_nbin);
        g0_wg.allocate(num_cols * nlayer_wg_nbin);

        g0_band.allocate(num_cols * nlayer_nbin);
        w0_band.allocate(num_cols * nlayer_nbin);

        delta_col_upper.allocate(num_cols * nlayer);
        delta_col_lower.allocate(num_cols * nlayer);
    }

    //  dev_T_int.allocate(ninterface);

    // column mass
    // TODO: computed by grid in helios, should be computed by alfrodull or comes from THOR?


    meanmolmass_lay.allocate(num_cols * nlayer);
    // scatter cross section layer and interface
    // those are shared for print out
    scatter_cross_section_lay.allocate(num_cols * nlayer_nbin);
    planckband_lay.allocate(num_cols * nlayer_plus2_nbin);
    opac_wg_lay.allocate(num_cols * nlayer_wg_nbin);

    if (!iso) {
        meanmolmass_int.allocate(num_cols * ninterface);

        scatter_cross_section_inter.allocate(num_cols * ninterface_nbin);
        planckband_int.allocate(num_cols * ninterface_nbin);
        opac_wg_int.allocate(num_cols * ninterface_wg_nbin);
    }


    if (iso) {
        trans_wg.allocate(num_cols * nlayer_wg_nbin);
    }
    else {
        trans_wg_upper.allocate(num_cols * nlayer_wg_nbin);
        trans_wg_lower.allocate(num_cols * nlayer_wg_nbin);
    }

    mu_star_cols.allocate(num_cols);
    hit_G_pm_denom_limit.allocate(1);

    // TODO: abstract this away into an interpolation class

    std::unique_ptr<double[]> weights = std::make_unique<double[]>(100);
    for (int i = 0; i < opacities.ny; i++)
        weights[i] = gauss_legendre_weights[opacities.ny - 1][i];

    gauss_weights.allocate(opacities.ny);
    gauss_weights.put(weights);

    USE_BENCHMARK();

#ifdef BENCHMARKING

    if (iso) {
        std::map<string, output_def> debug_arrays = {
            {"meanmolmass_lay",
             {meanmolmass_lay.ptr_ref(), nlayer, "meanmolmass_lay", "mmml", true, dummy}},
            {"planckband_lay",
             {planckband_lay.ptr_ref(), nlayer_plus2_nbin, "planckband_lay", "plkl", true, dummy}},
            {"planckband_int",
             {planckband_int.ptr_ref(), ninterface_nbin, "planckband_int", "plki", true, dummy}},
            {"opac_wg_lay",
             {opac_wg_lay.ptr_ref(), nlayer_wg_nbin, "opac_wg_lay", "opc", true, dummy}},
            {"trans_wg", {trans_wg.ptr_ref(), nlayer_wg_nbin, "trans_wg", "tr", true, dummy}},
            {"scat_cs_lay",
             {scatter_cross_section_lay.ptr_ref(),
              nlayer_nbin,
              "scat_cs_lay",
              "scsl",
              true,
              dummy}},
            {"delta_tau_wg",
             {delta_tau_wg.ptr_ref(), nlayer_wg_nbin, "delta_tau_wg", "dtw", true, dummy}},
            {"delta_colmass",
             {delta_col_mass.ptr_ref(), nlayer, "delta_colmass", "dcm", true, dummy}},
            {"M_term", {M_term.ptr_ref(), nlayer_wg_nbin, "M_term", "Mt", true, dummy}},
            {"N_term", {N_term.ptr_ref(), nlayer_wg_nbin, "N_term", "Nt", true, dummy}},
            {"P_term", {P_term.ptr_ref(), nlayer_wg_nbin, "P_term", "Pt", true, dummy}},
            {"G_plus", {G_plus.ptr_ref(), nlayer_wg_nbin, "G_plus", "Gp", true, dummy}},
            {"G_minus", {G_minus.ptr_ref(), nlayer_wg_nbin, "G_minus", "Gm", true, dummy}},
            {"w_0", {w0_wg.ptr_ref(), nlayer_wg_nbin, "w_0", "w0", true, dummy}},
            {"g_0", {g0_wg.ptr_ref(), (int)g0_wg.get_size(), "g_0", "g0", true, dummy}},

        };
        // TODO: add thomas algorithm variables - if needed

        /*
        A_buff.allocate(ninterface_wg_nbin * 4);       // thomas worker
        B_buff.allocate(ninterface_wg_nbin * 4);       // thomas worker
        C_buff.allocate(ninterface_wg_nbin * 4);       // thomas worker
        D_buff.allocate(ninterface_wg_nbin * 4);       // thomas worker
        C_prime_buff.allocate(ninterface_wg_nbin * 4); // thomas worker
        D_prime_buff.allocate(ninterface_wg_nbin * 4); // thomas worker
        X_buff.allocate(ninterface_wg_nbin * 4);       // thomas worker
	*/

        BENCH_POINT_REGISTER_PHY_VARS(debug_arrays, (), ());
    }
    else {
        std::map<string, output_def> debug_arrays = {
            {"meanmolmass_lay",
             {meanmolmass_lay.ptr_ref(), nlayer, "meanmolmass_lay", "mmml", true, dummy}},
            {"meanmolmass_int",
             {meanmolmass_int.ptr_ref(), ninterface, "meanmolmass_int", "mmmi", true, dummy}},
            {"planckband_lay",
             {planckband_lay.ptr_ref(), nlayer_plus2_nbin, "planckband_lay", "plkl", true, dummy}},
            {"planckband_int",
             {planckband_int.ptr_ref(), ninterface_nbin, "planckband_int", "plki", true, dummy}},
            {"opac_wg_lay",
             {opac_wg_lay.ptr_ref(), nlayer_wg_nbin, "opac_wg_lay", "opcl", true, dummy}},
            {"opac_wg_int",
             {opac_wg_int.ptr_ref(), ninterface_wg_nbin, "opac_wg_int", "opci", true, dummy}},
            {"trans_wg_upper",
             {trans_wg_upper.ptr_ref(), nlayer_wg_nbin, "trans_wg_upper", "tru", true, dummy}},
            {"trans_wg_lower",
             {trans_wg_lower.ptr_ref(), nlayer_wg_nbin, "trans_wg_lower", "trl", true, dummy}},
            {"scat_cs_lay",
             {scatter_cross_section_lay.ptr_ref(),
              nlayer_nbin,
              "scat_cs_lay",
              "scsl",
              true,
              dummy}},
            {"scat_cs_int",
             {scatter_cross_section_inter.ptr_ref(),
              ninterface_nbin,
              "scat_cs_int",
              "scsi",
              true,
              dummy}},
            {"delta_tau_wg_upper",
             {delta_tau_wg_upper.ptr_ref(),
              nlayer_wg_nbin,
              "delta_tau_wg_upper",
              "dtwu",
              true,
              dummy}},
            {"delta_tau_wg_lower",
             {delta_tau_wg_lower.ptr_ref(),
              nlayer_wg_nbin,
              "delta_tau_wg_lower",
              "dtwl",
              true,
              dummy}},

            {"delta_col_upper",
             {delta_col_upper.ptr_ref(), nlayer, "delta_col_upper", "dcu", true, dummy}},
            {"delta_col_lower",
             {delta_col_lower.ptr_ref(), nlayer, "delta_col_lower", "dcl", true, dummy}},
            {"M_upper", {M_upper.ptr_ref(), nlayer_wg_nbin, "M_upper", "Mu", true, dummy}},
            {"M_lower", {M_lower.ptr_ref(), nlayer_wg_nbin, "M_lower", "Ml", true, dummy}},
            {"N_upper", {N_upper.ptr_ref(), nlayer_wg_nbin, "N_upper", "Nu", true, dummy}},
            {"N_lower", {N_lower.ptr_ref(), nlayer_wg_nbin, "N_lower", "Nl", true, dummy}},
            {"P_upper", {P_upper.ptr_ref(), nlayer_wg_nbin, "P_upper", "Pu", true, dummy}},
            {"P_lower", {P_lower.ptr_ref(), nlayer_wg_nbin, "P_lower", "Pl", true, dummy}},
            {"G_plus_upper",
             {G_plus_upper.ptr_ref(), nlayer_wg_nbin, "G_plus_upper", "Gpu", true, dummy}},
            {"G_plus_lower",
             {G_plus_lower.ptr_ref(), nlayer_wg_nbin, "G_plus_lower", "Gpl", true, dummy}},
            {"G_minus_upper",
             {G_minus_upper.ptr_ref(), nlayer_wg_nbin, "G_minus_upper", "Gmu", true, dummy}},
            {"G_minus_lower",
             {G_minus_lower.ptr_ref(), nlayer_wg_nbin, "G_minus_lower", "Gml", true, dummy}},


            {"w_0_upper", {w0_wg_upper.ptr_ref(), nlayer_wg_nbin, "w_0_upper", "w0u", true, dummy}},
            {"w_0_lower", {w0_wg_lower.ptr_ref(), nlayer_wg_nbin, "w_0_lower", "w0l", true, dummy}},

        };
        BENCH_POINT_REGISTER_PHY_VARS(debug_arrays, (), ());
    }
#endif // BENCHMARKING
}

// set internal arrays to zero before loop
void alfrodull_engine::reset() {
    // delta tau, for weights. Only used internally (on device) for flux computations
    // and shared at the end for integration over wg
    if (iso) {
        delta_col_mass.zero();
        delta_tau_wg.zero();
        trans_wg.zero();
        M_term.zero();
        N_term.zero();
        P_term.zero();
        G_plus.zero();
        G_minus.zero();
        w0_wg.zero();
    }
    else {
        // noiso
        delta_tau_wg_upper.zero();
        delta_tau_wg_lower.zero();
        delta_col_upper.zero();
        delta_col_lower.zero();
        trans_wg_upper.zero();
        trans_wg_lower.zero();
        M_upper.zero();
        M_lower.zero();
        N_upper.zero();
        N_lower.zero();
        P_upper.zero();
        P_lower.zero();
        G_plus_upper.zero();
        G_plus_lower.zero();
        G_minus_upper.zero();
        G_minus_lower.zero();
        w0_wg_upper.zero();
        w0_wg_lower.zero();
    }

    dev_T_int.zero();

    meanmolmass_lay.zero();
    opac_wg_lay.zero();
    scatter_cross_section_lay.zero();
    planckband_lay.zero();

    if (!iso) {
        meanmolmass_int.zero();
        opac_wg_int.zero();
        scatter_cross_section_inter.zero();
        // planck function
        planckband_int.zero();
    }

    if (thomas) {
        A_buff.zero();       // thomas worker
        B_buff.zero();       // thomas worker
        C_buff.zero();       // thomas worker
        D_buff.zero();       // thomas worker
        C_prime_buff.zero(); // thomas worker
        D_prime_buff.zero(); // thomas worker
        X_buff.zero();       // thomas worker
    }
}

// return device pointers for helios data save
// TODO: how ugly can it get, really?
std::tuple<long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           long,
           int,
           int>
alfrodull_engine::get_device_pointers_for_helios_write() {
    return std::make_tuple((long)*scatter_cross_section_lay,
                           (long)*scatter_cross_section_inter,
                           (long)*opac_wg_lay,
                           (long)*planckband_lay,
                           (long)*planckband_int,
                           (long)*plancktable.planck_grid,
                           (long)*delta_tau_wg,
                           (long)*delta_tau_wg_upper,
                           (long)*delta_tau_wg_lower,
                           (long)*delta_col_mass,
                           (long)*delta_col_upper,
                           (long)*delta_col_lower,
                           (long)*meanmolmass_lay,
                           (long)*trans_wg,
                           (long)*trans_wg_upper,
                           (long)*trans_wg_lower,
                           plancktable.dim,
                           plancktable.step);
}

// get opacity data for helios
std::tuple<long, long, long, long, int, int> alfrodull_engine::get_opac_data_for_helios() {
    return std::make_tuple((long)*opacities.dev_opac_wave,
                           (long)*opacities.dev_opac_interwave,
                           (long)*opacities.dev_opac_deltawave,
                           (long)*opacities.dev_opac_y,
                           opacities.nbin,
                           opacities.ny);
}


// TODO: check how to enforce this: must be called after loading opacities and setting parameters
void alfrodull_engine::prepare_planck_table() {
    plancktable.construct_planck_table(
        *opacities.dev_opac_interwave, *opacities.dev_opac_deltawave, opacities.nbin, T_star);
}

void alfrodull_engine::correct_incident_energy(double* starflux_array_ptr,
                                               bool    real_star,
                                               bool    energy_budget_correction) {
    printf("T_star %g, energy budget_correction: %s\n",
           T_star,
           energy_budget_correction ? "true" : "false");
    if (T_star > 10 && energy_budget_correction) {
        dim3 grid((int(opacities.nbin) + 15) / 16, 1, 1);
        dim3 block(16, 1, 1);

        corr_inc_energy<<<grid, block>>>(*plancktable.planck_grid,
                                         starflux_array_ptr,
                                         *opacities.dev_opac_deltawave,
                                         real_star,
                                         opacities.nbin,
                                         T_star,
                                         plancktable.dim);

        cudaDeviceSynchronize();
    }

    // //nplanck_grid = (plancktable.dim+1)*opacities.nbin;
    // // print out planck grid for debug
    // std::unique_ptr<double[]> plgrd = std::make_unique<double[]>(plancktable.nplanck_grid);

    // plancktable.planck_grid.fetch(plgrd);
    // for (int i = 0; i < plancktable.nplanck_grid; i++)
    //   printf("array[%d] : %g\n", i, plgrd[i]);
}


void alfrodull_engine::set_z_calc_func(std::function<void()>& fun) {
    calc_z_func = fun;
}

void alfrodull_engine::call_z_callback() {
    if (calc_z_func)
        calc_z_func();
}

void alfrodull_engine::set_clouds_data(const bool& clouds_,
                                       double*     cloud_abs_cross_lay_,
                                       double*     cloud_abs_cross_int_,
                                       double*     cloud_scat_cross_lay_,
                                       double*     cloud_scat_cross_int_,
                                       double*     g_0_cloud_lay_,
                                       double*     g_0_cloud_int_,
                                       double      fcloud_) {
    // For now, cloud data are values per wavelength bins, with the input computed for the
    // correct wavelength bins, and used for the full column
    // so this is directly forwarded as "layer and interface" values,
    // for usage per volume, we need to allocate these arrays per volume
    // and an interpolation/lookup function from the input table to the volume
    // element parameters (P,T,...)  needs to be implemented, similar to opacity lookup.
    cloud_abs_cross_lay  = cloud_abs_cross_lay_;
    cloud_abs_cross_int  = cloud_abs_cross_int_;
    cloud_scat_cross_lay = cloud_scat_cross_lay_;
    cloud_scat_cross_int = cloud_scat_cross_int_;
    g_0_cloud_lay        = g_0_cloud_lay_;
    g_0_cloud_int        = g_0_cloud_int_;

    clouds = clouds_;
    fcloud = fcloud_;
}


// var already present:
// bool iso
void alfrodull_engine::compute_radiative_transfer(
    // prepare_compute_flux

    // TODO: planck value tabulated and then interpolated
    double* dev_starflux, // in: pil
    // state variables
    // TODO: check which ones can be internal only
    double*
                dev_T_lay_cols, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    double*     dev_T_int_cols, // in: it, pii, ioi, mmmi, kii
    double*     dev_p_lay_cols, // in: io, mmm, kil
    double*     dev_p_int_cols, // in: ioi, mmmi, kii
    const bool& interpolate_temp_and_pres,
    const bool& interp_and_calc_flux_step,
    double*     z_lay,
    bool        single_walk,
    double*     F_down_wg_cols,
    double*     F_up_wg_cols,
    double*     Fc_down_wg_cols,
    double*     Fc_up_wg_cols,
    double*     F_dir_wg_cols,
    double*     Fc_dir_wg_cols,
    double      delta_tau_limit,
    double*     F_down_tot_cols,
    double*     F_up_tot_cols,
    double*     F_dir_tot_cols,
    double*     F_net_cols,
    double*     F_down_band_cols,
    double*     F_up_band_cols,
    double*     F_dir_band_cols,
    double*     F_up_TOA_spectrum_cols,
    double*     zenith_angle_cols,
    int         num_cols,
    int         current_col_temp) // number of columns this function works on
{
    int nbin = opacities.nbin;
    int ny   = opacities.ny;
    USE_BENCHMARK();
    {
        prepare_compute_flux(
            dev_starflux,
            dev_T_lay_cols, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
            dev_T_int_cols,   // in: it, pii, ioi, mmmi, kii
            dev_p_lay_cols,   // in: io, mmm, kil
            dev_p_int_cols,   // in: ioi, mmmi, kii
            *opac_wg_lay,     // out: io
            *opac_wg_int,     // out: ioi
            *meanmolmass_lay, // out: mmm
            *meanmolmass_int, // out: mmmi
            real_star,        // pil
            fake_opac,        // io
            interpolate_temp_and_pres,
            interp_and_calc_flux_step,
            num_cols);

        cuda_check_status_or_exit(__FILE__, __LINE__);


        BENCH_POINT_I_S(debug_nstep,
                        debug_col_idx,
                        "Alf_prep_flx",
                        (),
                        ("opac_wg_lay",
                         "opac_wg_int",
                         "meanmolmass_lay",
                         "meanmolmass_int",
                         "cloud_scat_cross_lay",
                         "planckband_lay"));
    }
    double* deltalambda = *opacities.dev_opac_deltawave;


    // TODO: for development, propagate current column number
    //double* delta_colmass = &((*delta_col_mass)[c * nlayer]);
    // double* dev_T_lay     = &(dev_T_lay_cols[c * (nlayer + 1)]);
    // double* dev_T_int     = &(dev_T_int_cols[c * ninterface]);
    // double* dev_p_lay     = &(dev_p_lay_cols[c * nlayer]);
    // double* dev_p_int     = &(dev_p_int_cols[c * ninterface]);

    // double* F_down_wg    = &(F_down_wg_cols[c * ninterface * nbin * ny]);
    // double* F_up_wg      = &(F_up_wg_cols[c * ninterface * nbin * ny]);
    // double* Fc_down_wg   = &(Fc_down_wg_cols[c * ninterface * nbin * ny]);
    // double* Fc_up_wg     = &(Fc_up_wg_cols[c * ninterface * nbin * ny]);
    //double* zenith_angle = &(zenith_angle_cols[c]);

    //int column_offset_int = c * ninterface;


    // double* F_down_tot = &(F_down_tot_cols[column_offset_int]);
    // double* F_up_tot   = &(F_up_tot_cols[column_offset_int]);
    // double* F_dir_tot  = &(F_dir_tot_cols[column_offset_int]);
    // double* F_net      = &(F_net_cols[column_offset_int]);

    //            double* F_dir_band_col    = &((*F_dir_band)[ninterface * nbin]);
    // double* F_down_band = &(F_down_band_cols[c * ninterface * nbin]);
    // double* F_up_band   = &(F_up_band_cols[c * ninterface * nbin]);
    // double* F_dir_band  = &(F_dir_band_cols[c * ninterface * nbin]);

    // double* F_up_TOA_spectrum = &(F_up_TOA_spectrum_cols[c * nbin]);


    // also lookup and interpolate cloud values here if cloud values
    // per volume element is needed
    // fill in g_0_cloud_lay, g_0_cloud_int, cloud_scat_cross_lay, cloud_scat_cross_int
    if (interp_and_calc_flux_step) {
        if (iso) {
            BENCH_POINT_I_S(debug_nstep, debug_col_idx, "Alf_prep_II", (), ("delta_colmass"));

            calculate_transmission_iso(*trans_wg,            // out
                                       *delta_col_mass,      // in
                                       *opac_wg_lay,         // in
                                       cloud_abs_cross_lay,  // in
                                       *meanmolmass_lay,     // in
                                       cloud_scat_cross_lay, // in
                                       g_0_cloud_lay,        // in
                                       g_0,
                                       epsi,
                                       epsilon2,
                                       zenith_angle_cols,
                                       scat,
                                       clouds,
                                       num_cols);

            BENCH_POINT_I_S(debug_nstep,
                            debug_col_idx,
                            "Alf_comp_trans",
                            (),
                            ("delta_colmass",
                             "trans_wg",
                             "delta_tau_wg",
                             "M_term",
                             "N_term",
                             "P_term",
                             "w_0",
                             "g_0",
                             "G_plus",
                             "G_minus"));
        }
        else {
            BENCH_POINT_I_S(debug_nstep,
                            debug_col_idx,
                            "Alf_prep_II",
                            (),
                            ("delta_col_upper", "delta_col_lower", ));
            calculate_transmission_noniso(*trans_wg_upper,
                                          *trans_wg_lower,
                                          *delta_col_upper,
                                          *delta_col_lower,
                                          *opac_wg_lay,
                                          *opac_wg_int,
                                          cloud_abs_cross_lay,
                                          cloud_abs_cross_int,
                                          *meanmolmass_lay,
                                          *meanmolmass_int,
                                          cloud_scat_cross_lay,
                                          cloud_scat_cross_int,
                                          g_0_cloud_lay,
                                          g_0_cloud_int,
                                          g_0,
                                          epsi,
                                          epsilon2,
                                          zenith_angle_cols,
                                          scat,
                                          clouds,
                                          num_cols);
            BENCH_POINT_I_S(debug_nstep,
                            debug_col_idx,
                            "Alf_comp_trans",
                            (),
                            ("trans_wg_upper",
                             "trans_wg_lower",
                             "delta_tau_wg_upper",
                             "delta_tau_wg_lower",
                             "planckband_lay",
                             "planckband_int",
                             "M_upper",
                             "M_lower",
                             "N_upper",
                             "N_lower",
                             "P_upper",
                             "P_lower",
                             "G_plus_upper",
                             "G_plus_lower",
                             "G_minus_upper",
                             "G_minus_lower",
                             "w_0_upper",
                             "w_0_lower"));
        }

        cuda_check_status_or_exit(__FILE__, __LINE__);
        call_z_callback();

        direct_beam_flux(F_dir_wg_cols,
                         Fc_dir_wg_cols,
                         z_lay,
                         R_planet,
                         R_star,
                         a,
                         dir_beam,
                         geom_zenith_corr,
                         num_cols);

        BENCH_POINT_I_S(debug_nstep, debug_col_idx, "Alf_dir_beam_trans", (), ("F_dir_wg"));

        cuda_check_status_or_exit(__FILE__, __LINE__);
    }

    if (thomas) {
        if (iso) {
            populate_spectral_flux_iso_thomas(F_down_wg_cols, // out
                                              F_up_wg_cols,   // out
                                              F_dir_wg_cols,  // in
                                              *g0_wg,         // in
                                              single_walk,
                                              R_star,
                                              a,
                                              f_factor,
                                              epsi,
                                              w_0_limit,
                                              dir_beam,
                                              clouds,
                                              num_cols);
        }
        else {
            populate_spectral_flux_noniso_thomas(F_down_wg_cols,
                                                 F_up_wg_cols,
                                                 Fc_down_wg_cols,
                                                 Fc_up_wg_cols,
                                                 F_dir_wg_cols,
                                                 Fc_dir_wg_cols,
                                                 *g0_wg_upper,
                                                 *g0_wg_lower,
                                                 single_walk,
                                                 R_star,
                                                 a,
                                                 f_factor,
                                                 epsi,
                                                 w_0_limit,
                                                 delta_tau_limit,
                                                 dir_beam,
                                                 clouds,
                                                 *trans_wg_upper,
                                                 *trans_wg_lower,
                                                 num_cols);
        }
        cuda_check_status_or_exit(__FILE__, __LINE__);

        BENCH_POINT_I_S(
            debug_nstep, debug_col_idx, "Alf_pop_spec_flx_thomas", (), ("F_up_wg", "F_down_wg"));
    }
    else {
        int nscat_step = 0;
        if (single_walk)
            nscat_step = 200;
        else
            nscat_step = 3;

        if (!scat)
            nscat_step = 0;

        for (int scat_iter = 0; scat_iter < nscat_step + 1; scat_iter++) {
            if (iso) {
                populate_spectral_flux_iso(F_down_wg_cols, // out
                                           F_up_wg_cols,   // out
                                           F_dir_wg_cols,  // in
                                           *g0_wg,         // in
                                           single_walk,
                                           R_star,
                                           a,
                                           f_factor,
                                           epsi,
                                           w_0_limit,
                                           dir_beam,
                                           clouds,
                                           num_cols);
            }
            else {
                populate_spectral_flux_noniso(F_down_wg_cols,
                                              F_up_wg_cols,
                                              Fc_down_wg_cols,
                                              Fc_up_wg_cols,
                                              F_dir_wg_cols,
                                              Fc_dir_wg_cols,
                                              *g0_wg_upper,
                                              *g0_wg_lower,
                                              single_walk,
                                              R_star,
                                              a,
                                              f_factor,
                                              epsi,
                                              w_0_limit,
                                              delta_tau_limit,
                                              dir_beam,
                                              clouds,
                                              *trans_wg_upper,
                                              *trans_wg_lower,
                                              num_cols);
            }

            cuda_check_status_or_exit(__FILE__, __LINE__);
        }

        BENCH_POINT_I_S(
            debug_nstep, debug_col_idx, "Alf_pop_spec_flx", (), ("F_up_wg", "F_down_wg"));
    }


    double* gauss_weight = *gauss_weights;
    integrate_flux(deltalambda,
                   F_down_tot_cols,
                   F_up_tot_cols,
                   F_dir_tot_cols,
                   F_net_cols,
                   F_down_wg_cols,
                   F_up_wg_cols,
                   F_dir_wg_cols,
                   F_down_band_cols,
                   F_up_band_cols,
                   F_dir_band_cols,
                   F_up_TOA_spectrum_cols,
                   gauss_weight,
                   num_cols);

    BENCH_POINT_I_S(
        debug_nstep, debug_col_idx, "Alf_int_flx", (), ("F_up_band", "F_down_band", "F_dir_band"));


    cuda_check_status_or_exit(__FILE__, __LINE__);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("compute_radiative_transfer: cuda error: %s\n", cudaGetErrorString(err));
    }
}

bool alfrodull_engine::prepare_compute_flux(
    // TODO: planck value tabulated and then interpolated
    double* dev_starflux, // in: pil
    // state variables
    // TODO: check which ones can be internal only
    double*
                  dev_T_lay_cols, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    double*       dev_T_int_cols,           // in: it, pii, ioi, mmmi, kii
    double*       dev_p_lay_cols,           // in: io, mmm, kil
    double*       dev_p_int_cols,           // in: ioi, mmmi, kii
    double*       dev_opac_wg_lay_cols,     // out: io
    double*       dev_opac_wg_int_cols,     // out: ioi
    double*       dev_meanmolmass_lay_cols, // out: mmm
    double*       dev_meanmolmass_int_cols, // out: mmmi
    const bool&   real_star,                // pil
    const double& fake_opac,                // io
    const bool&   interpolate_temp_and_pres,
    const bool&   interp_and_calc_flux_step,
    const int&    num_cols) {

    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    // TODO: check where those planckband values are used, where used here in
    // calculate_surface_planck and correc_surface_emission that's not used anymore
    // out: csp, cse
    int plancktable_dim  = plancktable.dim;
    int plancktable_step = plancktable.step;

    if (interpolate_temp_and_pres) {
        // it
        dim3 it_grid(int((ninterface + 15) / 16), 1, 1);
        dim3 it_block(16, 1, 1);

        interpolate_temperature<<<it_grid, it_block>>>(dev_T_lay_cols, // out
                                                       dev_T_int_cols, // in
                                                       ninterface);
        cudaDeviceSynchronize();
    }


    // pil
    dim3 pil_grid(int((nbin + 15) / 16), int(((nlayer + 2) + 15)) / 16, num_cols);
    dim3 pil_block(16, 16, 1);
    planck_interpol_layer<<<pil_grid, pil_block>>>(dev_T_lay_cols,           // in
                                                   *planckband_lay,          // out
                                                   *plancktable.planck_grid, // in
                                                   dev_starflux,             // in
                                                   real_star,
                                                   nlayer,
                                                   nbin,
                                                   plancktable_dim,
                                                   plancktable_step);
    cudaDeviceSynchronize();

    if (!iso) {
        // pii
        dim3 pii_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), num_cols);
        dim3 pii_block(16, 16, 1);
        planck_interpol_interface<<<pii_grid, pii_block>>>(dev_T_int_cols,           // in
                                                           *planckband_int,          // out
                                                           *plancktable.planck_grid, // in
                                                           ninterface,
                                                           nbin,
                                                           plancktable_dim,
                                                           plancktable_step);
        cudaDeviceSynchronize();
    }

    if (interp_and_calc_flux_step) {
        // io
        dim3 io_grid(int((nbin + 15) / 16), int((nlayer + 15) / 16), num_cols);
        dim3 io_block(16, 16, 1);
        // TODO: should move fake_opac (opacity limit somewhere into opacity_table/interpolation component?)
        // out -> opacities (dev_opac_wg_lay)
        // out -> scetter cross section (scatter_cross_section_...)
        interpolate_opacities<<<io_grid, io_block>>>(dev_T_lay_cols,                     // in
                                                     *opacities.dev_temperatures,        // in
                                                     dev_p_lay_cols,                     // in
                                                     *opacities.dev_pressures,           // in
                                                     *opacities.dev_kpoints,             // in
                                                     dev_opac_wg_lay_cols,               // out
                                                     *opacities.dev_scat_cross_sections, // in
                                                     *scatter_cross_section_lay,         // out
                                                     opacities.n_pressures,
                                                     opacities.n_temperatures,
                                                     opacities.ny,
                                                     nbin,
                                                     fake_opac,
                                                     nlayer + 1,
                                                     nlayer,
                                                     nlayer);


        cudaDeviceSynchronize();

        if (!iso) {

            // ioi
            dim3 ioi_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), num_cols);
            dim3 ioi_block(16, 16, 1);

            interpolate_opacities<<<ioi_grid, ioi_block>>>(dev_T_int_cols,              // in
                                                           *opacities.dev_temperatures, // in
                                                           dev_p_int_cols,              // in
                                                           *opacities.dev_pressures,    // in
                                                           *opacities.dev_kpoints,      // in
                                                           dev_opac_wg_int_cols,        // out
                                                           *opacities.dev_scat_cross_sections, // in
                                                           *scatter_cross_section_inter, // out
                                                           opacities.n_pressures,
                                                           opacities.n_temperatures,
                                                           opacities.ny,
                                                           nbin,
                                                           fake_opac,
                                                           ninterface,
                                                           ninterface,
                                                           ninterface);

            cudaDeviceSynchronize();
        }
        // mmm
        dim3 mmm_grid(int((nlayer + 15) / 16), 1, num_cols);
        dim3 mmm_block(16, 1, 1);

        meanmolmass_interpol<<<mmm_grid, mmm_block>>>(dev_T_lay_cols,              // in
                                                      *opacities.dev_temperatures, // in
                                                      dev_meanmolmass_lay_cols,    // out
                                                      *opacities.dev_meanmolmass,  // in
                                                      dev_p_lay_cols,              // in
                                                      *opacities.dev_pressures,    // in
                                                      opacities.n_pressures,
                                                      opacities.n_temperatures,
                                                      nlayer + 1,
                                                      nlayer,
                                                      nlayer);


        cudaDeviceSynchronize();

        if (!iso) {
            // mmmi
            dim3 mmmi_grid(int((ninterface + 15) / 16), 1, num_cols);
            dim3 mmmi_block(16, 1, 1);

            meanmolmass_interpol<<<mmmi_grid, mmmi_block>>>(dev_T_int_cols,              // in
                                                            *opacities.dev_temperatures, // in
                                                            dev_meanmolmass_int_cols,    // out
                                                            *opacities.dev_meanmolmass,  // in
                                                            dev_p_int_cols,              // in
                                                            *opacities.dev_pressures,    // in
                                                            opacities.n_pressures,
                                                            opacities.n_temperatures,
                                                            ninterface,
                                                            ninterface,
                                                            ninterface);


            cudaDeviceSynchronize();
        }
    }

    // TODO: add state check and return value

    return true;
}


void alfrodull_engine::integrate_flux(double* deltalambda,
                                      double* F_down_tot,
                                      double* F_up_tot,
                                      double* F_dir_tot,
                                      double* F_net,
                                      double* F_down_wg,
                                      double* F_up_wg,
                                      double* F_dir_wg,
                                      double* F_down_band,
                                      double* F_up_band,
                                      double* F_dir_band,
                                      double* F_up_TOA_spectrum,
                                      double* gauss_weight,
                                      int     num_cols) {

    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    {
        int  num_levels_per_block = 16;
        int  num_bins_per_block   = 16;
        dim3 gridsize(
            ninterface / num_levels_per_block + 1, nbin / num_bins_per_block + 1, num_cols);
        dim3 blocksize(num_levels_per_block, num_bins_per_block, 1);
        //printf("nbin: %d, ny: %d\n", nbin, ny);

        integrate_flux_band<<<gridsize, blocksize>>>(F_down_wg,
                                                     F_up_wg,
                                                     F_dir_wg,
                                                     F_down_band,
                                                     F_up_band,
                                                     F_dir_band,
                                                     F_up_TOA_spectrum,
                                                     gauss_weight,
                                                     nbin,
                                                     ninterface,
                                                     ny);

        cudaDeviceSynchronize();
    }

    {
        int  num_levels_per_block = 256;
        dim3 gridsize(ninterface / num_levels_per_block + 1, 1, num_cols);
        dim3 blocksize(num_levels_per_block, 1, 1);
        integrate_flux_tot<<<gridsize, blocksize>>>(deltalambda,
                                                    F_down_tot,
                                                    F_up_tot,
                                                    F_dir_tot,
                                                    F_net,
                                                    F_down_band,
                                                    F_up_band,
                                                    F_dir_band,
                                                    nbin,
                                                    ninterface);
        cudaDeviceSynchronize();
    }
}

void alfrodull_engine::calculate_transmission_iso(double* trans_wg,             // out
                                                  double* delta_colmass,        // in
                                                  double* opac_wg_lay,          // in
                                                  double* cloud_abs_cross_lay_, // in
                                                  double* meanmolmass_lay,      // in
                                                  double* cloud_scat_cross_lay, // in
                                                  double* g_0_cloud_lay_,       // in
                                                  double  g_0,
                                                  double  epsi,
                                                  double  epsilon2_,
                                                  double* zenith_angle_cols,
                                                  bool    scat,
                                                  bool    clouds,
                                                  int     num_cols) {
    bool hit_G_pm_denom_limit_h = false;
    // set global wiggle checker to 0;
    cudaMemcpy(
        *hit_G_pm_denom_limit, &hit_G_pm_denom_limit_h, sizeof(bool), cudaMemcpyHostToDevice);

    int nbin = opacities.nbin;

    int ny = opacities.ny;


    dim3 grid((nbin + 15) / 16, (num_cols * ny + 3) / 4, (nlayer + 3) / 4);
    dim3 block(16, 4, 4);
    trans_iso<<<grid, block>>>(trans_wg,
                               *delta_tau_wg,
                               *M_term,
                               *N_term,
                               *P_term,
                               *G_plus,
                               *G_minus,
                               delta_colmass,
                               opac_wg_lay,
                               cloud_abs_cross_lay_,
                               meanmolmass_lay,
                               *scatter_cross_section_lay,
                               cloud_scat_cross_lay,
                               *w0_wg,
                               *g0_wg,
                               g_0_cloud_lay_,
                               g_0,
                               epsi,
                               epsilon2_,
                               zenith_angle_cols,
                               *mu_star_cols,
                               w_0_limit,
                               scat,
                               nbin,
                               ny,
                               nlayer,
                               num_cols,
                               fcloud,
                               clouds,
                               scat_corr,
                               mu_star_wiggle_increment,
                               G_pm_limiter,
                               G_pm_denom_limit,
                               *hit_G_pm_denom_limit,
                               debug,
                               i2s_transition);

    cudaDeviceSynchronize();
    cudaMemcpy(
        &hit_G_pm_denom_limit_h, *hit_G_pm_denom_limit, sizeof(bool), cudaMemcpyDeviceToHost);

    if (hit_G_pm_denom_limit_h) {
        printf("Hit G_pm denom limit, wiggled mu_star\n");
    }
}

void alfrodull_engine::calculate_transmission_noniso(double* trans_wg_upper,
                                                     double* trans_wg_lower,
                                                     double* delta_col_upper,
                                                     double* delta_col_lower,
                                                     double* opac_wg_lay,
                                                     double* opac_wg_int,
                                                     double* cloud_abs_cross_lay_,
                                                     double* cloud_abs_cross_int_,
                                                     double* meanmolmass_lay,
                                                     double* meanmolmass_int,
                                                     double* cloud_scat_cross_lay,
                                                     double* cloud_scat_cross_int,
                                                     double* g_0_cloud_lay_,
                                                     double* g_0_cloud_int_,
                                                     double  g_0,
                                                     double  epsi,
                                                     double  epsilon2_,
                                                     double* zenith_angle_cols,
                                                     bool    scat,
                                                     bool    clouds,
                                                     int     num_cols) {

    bool hit_G_pm_denom_limit_h = false;

    // set wiggle checker to 0;
    cudaMemcpy(
        *hit_G_pm_denom_limit, &hit_G_pm_denom_limit_h, sizeof(bool), cudaMemcpyHostToDevice);

    int nbin = opacities.nbin;

    int ny = opacities.ny;

    dim3 grid((nbin + 15) / 16, (num_cols * ny + 3) / 4, (nlayer + 3) / 4);
    dim3 block(16, 4, 4);

    trans_noniso<<<grid, block>>>(trans_wg_upper,
                                  trans_wg_lower,
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
                                  delta_col_upper,
                                  delta_col_lower,
                                  opac_wg_lay,
                                  opac_wg_int,
                                  cloud_abs_cross_lay_,
                                  cloud_abs_cross_int_,
                                  meanmolmass_lay,
                                  meanmolmass_int,
                                  *scatter_cross_section_lay,
                                  *scatter_cross_section_inter,
                                  cloud_scat_cross_lay,
                                  cloud_scat_cross_int,
                                  *w0_wg_upper,
                                  *w0_wg_lower,
                                  *g0_wg_upper,
                                  *g0_wg_lower,
                                  g_0_cloud_lay_,
                                  g_0_cloud_int_,
                                  g_0,
                                  epsi,
                                  epsilon2_,
                                  zenith_angle_cols,
                                  *mu_star_cols,
                                  w_0_limit,
                                  scat,
                                  nbin,
                                  ny,
                                  nlayer,
                                  num_cols,
                                  fcloud,
                                  clouds,
                                  scat_corr,
                                  mu_star_wiggle_increment,
                                  G_pm_limiter,
                                  G_pm_denom_limit,
                                  *hit_G_pm_denom_limit,
                                  debug,
                                  i2s_transition);
    cudaDeviceSynchronize();
    cudaMemcpy(
        &hit_G_pm_denom_limit_h, *hit_G_pm_denom_limit, sizeof(bool), cudaMemcpyDeviceToHost);

    if (hit_G_pm_denom_limit_h) {
        printf("Hit G_pm denom limit, wiggled mu_star\n");
    }
}

bool alfrodull_engine::direct_beam_flux(double* F_dir_wg,
                                        double* Fc_dir_wg,
                                        double* z_lay,
                                        double  R_planet,
                                        double  R_star,
                                        double  a,
                                        bool    dir_beam,
                                        bool    geom_zenith_corr,
                                        int     num_cols) {

    int nbin = opacities.nbin;

    int ny = opacities.ny;

    //printf("R_star: %g, R_planet: %g, a: %g\n", R_star, R_planet, a);
    //printf("dir beam: %d, geom_z_corr: %d, mu_star: %g\n", dir_beam, geom_zenith_corr, mu_star);
    if (iso) {
        dim3 grid((ninterface + 3) / 4, (nbin + 31) / 32, (num_cols * ny + 3) / 4);
        dim3 block(4, 32, 4);
        fdir_iso<<<grid, block>>>(F_dir_wg,
                                  *planckband_lay,
                                  *delta_tau_wg,
                                  z_lay,
                                  *mu_star_cols,
                                  mu_star_limit,
                                  R_planet,
                                  R_star,
                                  a,
                                  dir_beam,
                                  geom_zenith_corr,
                                  ninterface,
                                  nbin,
                                  ny,
                                  num_cols);

        cudaDeviceSynchronize();
    }
    else {
        dim3 grid((ninterface + 3) / 4, (nbin + 31) / 32, (num_cols * ny + 3) / 4);
        dim3 block(4, 32, 4);
        fdir_noniso<<<grid, block>>>(F_dir_wg,
                                     Fc_dir_wg,
                                     *planckband_lay,
                                     *delta_tau_wg_upper,
                                     *delta_tau_wg_lower,
                                     z_lay,
                                     *mu_star_cols,
                                     mu_star_limit,
                                     R_planet,
                                     R_star,
                                     a,
                                     dir_beam,
                                     geom_zenith_corr,
                                     ninterface,
                                     nbin,
                                     ny,
                                     num_cols);

        cudaDeviceSynchronize();
    }

    return true;
}

bool alfrodull_engine::populate_spectral_flux_iso_thomas(double* F_down_wg, // out
                                                         double* F_up_wg,   // out
                                                         double* F_dir_wg,  // in
                                                         double* g_0_tot,   // in
                                                         bool    singlewalk,
                                                         double  Rstar,
                                                         double  a,
                                                         double  f_factor,
                                                         double  epsi,
                                                         double  w_0_limit,
                                                         bool    dir_beam,
                                                         bool    clouds,
                                                         int     num_cols) {

    int nbin = opacities.nbin;
    int ny   = opacities.ny;


    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, num_cols);
    dim3 block(16, 16, 1);
    fband_iso_thomas<<<grid, block>>>(F_down_wg,
                                      F_up_wg,
                                      F_dir_wg,
                                      *planckband_lay,
                                      *w0_wg,
                                      *M_term,
                                      *N_term,
                                      *P_term,
                                      *G_plus,
                                      *G_minus,
                                      *A_buff,       // thomas worker
                                      *B_buff,       // thomas worker
                                      *C_buff,       // thomas worker
                                      *D_buff,       // thomas worker
                                      *C_prime_buff, // thomas worker
                                      *D_prime_buff, // thomas worker
                                      *X_buff,       // thomas worker
                                      g_0_tot,
                                      singlewalk,
                                      Rstar,
                                      a,
                                      ninterface,
                                      nbin,
                                      f_factor,
                                      *mu_star_cols,
                                      ny,
                                      num_cols,
                                      epsi,
                                      dir_beam,
                                      clouds,
                                      scat_corr,
                                      debug,
                                      i2s_transition);

    cudaDeviceSynchronize();

    return true;
}

bool alfrodull_engine::populate_spectral_flux_iso(double* F_down_wg, // out
                                                  double* F_up_wg,   // out
                                                  double* F_dir_wg,  // in
                                                  double* g_0_tot,   // in
                                                  bool    singlewalk,
                                                  double  Rstar,
                                                  double  a,
                                                  double  f_factor,
                                                  double  epsi,
                                                  double  w_0_limit,
                                                  bool    dir_beam,
                                                  bool    clouds,
                                                  int     num_cols) {

    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, num_cols);
    dim3 block(16, 16, 1);
    fband_iso_notabu<<<grid, block>>>(F_down_wg,
                                      F_up_wg,
                                      F_dir_wg,
                                      *planckband_lay,
                                      *w0_wg,
                                      *M_term,
                                      *N_term,
                                      *P_term,
                                      *G_plus,
                                      *G_minus,
                                      g_0_tot,
                                      singlewalk,
                                      Rstar,
                                      a,
                                      ninterface,
                                      nbin,
                                      f_factor,
                                      *mu_star_cols,
                                      ny,
                                      num_cols,
                                      epsi,
                                      dir_beam,
                                      clouds,
                                      scat_corr,
                                      debug,
                                      i2s_transition);

    cudaDeviceSynchronize();
    return true;
}

// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
bool alfrodull_engine::populate_spectral_flux_noniso(double* F_down_wg,
                                                     double* F_up_wg,
                                                     double* Fc_down_wg,
                                                     double* Fc_up_wg,
                                                     double* F_dir_wg,
                                                     double* Fc_dir_wg,
                                                     double* g_0_tot_upper,
                                                     double* g_0_tot_lower,
                                                     bool    singlewalk,
                                                     double  Rstar,
                                                     double  a,
                                                     double  f_factor,
                                                     double  epsi,
                                                     double  w_0_limit,
                                                     double  delta_tau_limit,
                                                     bool    dir_beam,
                                                     bool    clouds,
                                                     double* trans_wg_upper,
                                                     double* trans_wg_lower,
                                                     int     num_cols) {
    int nbin = opacities.nbin;
    int ny   = opacities.ny;


    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, num_cols);
    dim3 block(16, 16, 1);
    // calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
    fband_noniso_notabu<<<grid, block>>>(F_down_wg,
                                         F_up_wg,
                                         Fc_down_wg,
                                         Fc_up_wg,
                                         F_dir_wg,
                                         Fc_dir_wg,
                                         *planckband_lay,
                                         *planckband_int,
                                         *w0_wg_upper,
                                         *w0_wg_lower,
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
                                         g_0_tot_upper,
                                         g_0_tot_lower,
                                         singlewalk,
                                         Rstar,
                                         a,
                                         ninterface,
                                         nbin,
                                         f_factor,
                                         *mu_star_cols,
                                         ny,
                                         num_cols,
                                         epsi,
                                         delta_tau_limit,
                                         dir_beam,
                                         clouds,
                                         scat_corr,
                                         debug,
                                         i2s_transition);

    cudaDeviceSynchronize();

    return true;
}

// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
bool alfrodull_engine::populate_spectral_flux_noniso_thomas(double* F_down_wg,
                                                            double* F_up_wg,
                                                            double* Fc_down_wg,
                                                            double* Fc_up_wg,
                                                            double* F_dir_wg,
                                                            double* Fc_dir_wg,
                                                            double* g_0_tot_upper,
                                                            double* g_0_tot_lower,
                                                            bool    singlewalk,
                                                            double  Rstar,
                                                            double  a,
                                                            double  f_factor,
                                                            double  epsi,
                                                            double  w_0_limit,
                                                            double  delta_tau_limit,
                                                            bool    dir_beam,
                                                            bool    clouds,
                                                            double* trans_wg_upper,
                                                            double* trans_wg_lower,
                                                            int     num_cols) {
    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    dim3 block(16, 16, 1);

    dim3 grid((nbin + 15) / 16, (ny + 15) / 16, num_cols);

    // calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
    fband_noniso_thomas<<<grid, block>>>(F_down_wg,
                                         F_up_wg,
                                         Fc_down_wg,
                                         Fc_up_wg,
                                         F_dir_wg,
                                         Fc_dir_wg,
                                         *planckband_lay,
                                         *planckband_int,
                                         *w0_wg_upper,
                                         *w0_wg_lower,
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
                                         *A_buff,       // thomas worker
                                         *B_buff,       // thomas worker
                                         *C_buff,       // thomas worker
                                         *D_buff,       // thomas worker
                                         *C_prime_buff, // thomas worker
                                         *D_prime_buff, // thomas worker
                                         *X_buff,       // thomas worker
                                         g_0_tot_upper,
                                         g_0_tot_lower,
                                         singlewalk,
                                         Rstar,
                                         a,
                                         ninterface,
                                         nbin,
                                         f_factor,
                                         *mu_star_cols,
                                         ny,
                                         1,
                                         epsi,
                                         delta_tau_limit,
                                         dir_beam,
                                         clouds,
                                         scat_corr,
                                         debug,
                                         i2s_transition);

    cudaDeviceSynchronize();

    return true;
}

bool alfrodull_engine::get_column_integrated_g0_w0(double* g0_, double* w0_) {

    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    if (!iso) {
        // compute mean of upper and lower band
        int  num_val              = nlayer * nbin * ny;
        int  num_levels_per_block = 256;
        dim3 gridsize(num_val / num_levels_per_block + 1);
        dim3 blocksize(num_levels_per_block);

        arrays_mean<<<gridsize, blocksize>>>(*w0_wg_upper, *w0_wg_lower, *w0_wg, num_val);
        arrays_mean<<<gridsize, blocksize>>>(*g0_wg_upper, *g0_wg_lower, *g0_wg, num_val);
    }

    {
        int  num_levels_per_block = 16;
        int  num_bins_per_block   = 16;
        dim3 gridsize(nlayer / num_levels_per_block + 1, nbin / num_bins_per_block + 1);
        dim3 blocksize(num_levels_per_block, num_bins_per_block);

        integrate_val_band<<<gridsize, blocksize>>>(*w0_wg, w0_, *gauss_weights, nbin, nlayer, ny);
        integrate_val_band<<<gridsize, blocksize>>>(*g0_wg, g0_, *gauss_weights, nbin, nlayer, ny);

        cudaDeviceSynchronize();
    }

    // {
    //     int  num_levels_per_block = 256;
    //     dim3 gridsize(ninterface / num_levels_per_block + 1);
    //     dim3 blocksize(num_levels_per_block);

    //     double* deltalambda = *opacities.dev_opac_deltawave;

    //     integrate_val_tot<<<gridsize, blocksize>>>(g0_, *g0_band, deltalambda, nbin, nlayer);
    //     integrate_val_tot<<<gridsize, blocksize>>>(w0_, *w0_band, deltalambda, nbin, nlayer);

    //     cudaDeviceSynchronize();
    // }
    return true;
}
