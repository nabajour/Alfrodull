#include "alfrodull_engine.h"
#include "gauss_legendre_weights.h"

#include "calculate_physics.h"
#include "integrate_flux.h"
#include "interpolate_values.h"


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
                                      const double& T_surf_,
                                      const double& surf_albedo_,
                                      const double& g_0_,
                                      const double& epsi_,
                                      const double& mu_star_,
                                      const bool&   scat_,
                                      const bool&   scat_corr_,
                                      const double& R_planet_,
                                      const double& R_star_,
                                      const double& a_,
                                      const bool&   dir_beam_,
                                      const bool&   geom_zenith_corr_,
                                      const double& f_factor_,
                                      const double& w_0_limit_,
                                      const double& albedo_,
                                      const double& i2s_transition_,
                                      const bool&   debug_) {
    nlayer     = nlayer_;
    ninterface = nlayer + 1;
    iso        = iso_;
    T_star     = T_star_;

    real_star        = real_star_;
    fake_opac        = fake_opac_;
    T_surf           = T_surf_;
    surf_albedo      = surf_albedo_;
    g_0              = g_0_;
    epsi             = epsi_;
    mu_star          = mu_star_;
    scat             = scat_;
    scat_corr        = scat_corr_;
    R_planet         = R_planet_;
    R_star           = R_star_;
    a                = a_;
    dir_beam         = dir_beam_;
    geom_zenith_corr = geom_zenith_corr_;
    f_factor         = f_factor_;
    w_0_limit        = w_0_limit_;
    albedo           = albedo_;

    i2s_transition = i2s_transition;
    debug          = debug_;
    // TODO: maybe should stay in opacities object
    nbin = opacities.nbin;

    // prepare_planck_table();
}

void alfrodull_engine::allocate_internal_variables() {
    int nlayer_nbin        = nlayer * opacities.nbin;
    int nlayer_plus2_nbin  = (nlayer + 2) * opacities.nbin;
    int ninterface_nbin    = ninterface * opacities.nbin;
    int nlayer_wg_nbin     = nlayer * opacities.ny * opacities.nbin;
    int ninterface_wg_nbin = ninterface * opacities.ny * opacities.nbin;

    // scatter cross section layer and interface
    // those are shared for print out

    scatter_cross_section_lay.allocate(nlayer_nbin);
    scatter_cross_section_inter.allocate(ninterface_nbin);
    planckband_lay.allocate(nlayer_plus2_nbin);
    planckband_int.allocate(ninterface_nbin);


    if (iso) {
        delta_tau_wg.allocate(nlayer_wg_nbin);
    }
    else {
        delta_tau_wg_upper.allocate(nlayer_wg_nbin);
        delta_tau_wg_lower.allocate(nlayer_wg_nbin);
    }

    // flux computation internal quantities
    // TODO: not needed to allocate everything, depending on iso or noniso
    if (iso) {
        M_term.allocate(nlayer_wg_nbin);
        N_term.allocate(nlayer_wg_nbin);
        P_term.allocate(nlayer_wg_nbin);
        G_plus.allocate(nlayer_wg_nbin);
        G_minus.allocate(nlayer_wg_nbin);
        w_0.allocate(nlayer_wg_nbin);
    }
    else {
        M_upper.allocate(nlayer_wg_nbin);
        M_lower.allocate(nlayer_wg_nbin);
        N_upper.allocate(nlayer_wg_nbin);
        N_lower.allocate(nlayer_wg_nbin);
        P_upper.allocate(nlayer_wg_nbin);
        P_lower.allocate(nlayer_wg_nbin);
        G_plus_upper.allocate(nlayer_wg_nbin);
        G_plus_lower.allocate(nlayer_wg_nbin);
        G_minus_upper.allocate(nlayer_wg_nbin);
        G_minus_lower.allocate(nlayer_wg_nbin);
        w_0_upper.allocate(nlayer_wg_nbin);
        w_0_lower.allocate(nlayer_wg_nbin);
    }

    //  dev_T_int.allocate(ninterface);

    // column mass
    // TODO: computed by grid in helios, should be computed by alfrodull or comes from THOR?
    delta_col_mass.allocate(nlayer);
    delta_col_upper.allocate(nlayer);
    delta_col_lower.allocate(nlayer);


    meanmolmass_lay.allocate(nlayer);
    meanmolmass_int.allocate(ninterface);

    opac_wg_lay.allocate(nlayer_wg_nbin);

    trans_wg.allocate(nlayer_wg_nbin);

    if (!iso) {
        meanmolmass_lay.allocate(ninterface); // TODO: needs copy back to host
        opac_wg_int.allocate(ninterface_wg_nbin);
        trans_wg_upper.allocate(nlayer_wg_nbin);
        trans_wg_lower.allocate(nlayer_wg_nbin);
    }

    // TODO: abstract this away into an interpolation class

    std::unique_ptr<double[]> weights = std::make_unique<double[]>(100);
    for (int i = 0; i < opacities.ny; i++)
        weights[i] = gauss_legendre_weights[opacities.ny - 1][i];

    gauss_weights.allocate(opacities.ny);
    gauss_weights.put(weights);
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
                                       double*     cloud_opac_lay_,
                                       double*     cloud_opac_int_,
                                       double*     cloud_scat_cross_lay_,
                                       double*     cloud_scat_cross_int_,
                                       double*     g_0_tot_lay_,
                                       double*     g_0_tot_int_) {
    cloud_opac_lay       = cloud_opac_lay_;
    cloud_opac_int       = cloud_opac_int_;
    cloud_scat_cross_lay = cloud_scat_cross_lay_;
    cloud_scat_cross_int = cloud_scat_cross_int_;
    g_0_tot_lay          = g_0_tot_lay_;
    g_0_tot_int          = g_0_tot_int_;

    clouds = clouds_;
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
                dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    double*     dev_T_int, // in: it, pii, ioi, mmmi, kii
    double*     dev_p_lay, // in: io, mmm, kil
    double*     dev_p_int, // in: ioi, mmmi, kii
    const bool & interpolate_temp_and_pres, 
    const bool& interp_and_calc_flux_step,
    // calculate_transmission_iso
    //double* trans_wg,        // out
    //double* opac_wg_lay,     // in
    //double* cloud_opac_lay,  // in
    //double* meanmolmass_lay, // in
    //double* cloud_scat_cross_lay, // in
    //double* g_0_tot_lay,       // in
    //double  g_0,
    //double  epsi,
    //double  mu_star,
    //int     scat,
    //int     ny,
    //int     clouds,
    //int     scat_corr


    // calculate_transmission_non_iso
    // double* trans_wg,        // out
    // double* trans_wg_upper,
    // double* trans_wg_lower,
    // double* cloud_opac_lay,
    // double* cloud_opac_int,
    // double* cloud_scat_cross_lay,
    // double* cloud_scat_cross_int,
    // double* g_0_tot_lay,
    // double* g_0_tot_int,
    // double  g_0,
    // double  epsi,
    // double  mu_star,
    // int     scat,
    // int     clouds,
    // int     scat_corr,
    // direct_beam_flux
    //double* F_dir_wg,
    //double* Fc_dir_wg,
    double* z_lay,
    //double  mu_star,
    // double R_planet,
    // double R_star,
    // double a,
    // int    dir_beam,
    // int    geom_zenith_corr,
    //int     ny

    // spectral flux loop
    bool single_walk,
    // int scat_val, -> same as scat


    // populate_spectral_flux_iso
    //double* F_down_wg,    // out
    //double* F_up_wg,      // out
    //double* F_dir_wg,     // in
    //double* g_0_tot_lay,   // in
    //double  g_0,
    // int     singlewalk, -> single_walk
    //                                double  Rstar, -> R_star
    //double  a,
    // int     numinterfaces, -> ninterface
    // double f_factor,
    //double  mu_star,
    //int     ny,
    //double  epsi,
    // double w_0_limit,
    // double albedo,
    // populate_spectral_flux_noniso
    double* F_down_wg,
    double* F_up_wg,
    double* Fc_down_wg,
    double* Fc_up_wg,
    double* F_dir_wg,
    double* Fc_dir_wg,
    //double* g_0_tot_lay,
    //double* g_0_tot_int,
    //double  g_0,
    //int     singlewalk,
    //double  Rstar,
    //double  a,
    //int     numinterfaces,
    //double  f_factor,
    //double  mu_star,
    //int     ny,
    //double  epsi,
    //double  w_0_limit,
    double delta_tau_limit,
    //int     dir_beam,
    //int     clouds,
    //double  albedo, -> surf_albedo
    // double* trans_wg_upper,
    // double* trans_wg_lower,

    // integrate_flux
    double* F_down_tot,
    double* F_up_tot,
    double* F_net,
    //double* F_down_wg,
    //double* F_up_wg,
    //double* F_dir_wg,
    double* F_down_band,
    double* F_up_band,
    double* F_dir_band,
    // double* gauss_weight
    //int num_interfaces, -> ninterface
    //int ny
    double mu_star
) {

    double* delta_colmass = *delta_col_mass;


    double* deltalambda = *opacities.dev_opac_deltawave;

    prepare_compute_flux(
        dev_starflux,
        dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
        dev_T_int,        // in: it, pii, ioi, mmmi, kii
        dev_p_lay,        // in: io, mmm, kil
        dev_p_int,        // in: ioi, mmmi, kii
        *opac_wg_lay,     // out: io
        *opac_wg_int,     // out: ioi
        *meanmolmass_lay, // out: mmm
        *meanmolmass_int, // out: mmmi
        real_star,        // pil
        fake_opac,        // io
        T_surf,           // csp, cse, pil
        surf_albedo,      // cse
	interpolate_temp_and_pres, 
        interp_and_calc_flux_step);

    cuda_check_status_or_exit(__FILE__, __LINE__);


    if (interp_and_calc_flux_step) {
        if (iso) {
            calculate_transmission_iso(*trans_wg,            // out
                                       delta_colmass,        // in
                                       *opac_wg_lay,         // in
                                       cloud_opac_lay,       // in
                                       *meanmolmass_lay,     // in
                                       cloud_scat_cross_lay, // in
                                       g_0_tot_lay,          // in
                                       g_0,
                                       epsi,
                                       mu_star,
                                       scat,
                                       clouds);
        }
        else {
            calculate_transmission_noniso(*trans_wg_upper,
                                          *trans_wg_lower,
                                          *delta_col_upper,
                                          *delta_col_lower,
                                          *opac_wg_lay,
                                          *opac_wg_int,
                                          cloud_opac_lay,
                                          cloud_opac_int,
                                          *meanmolmass_lay,
                                          *meanmolmass_int,
                                          cloud_scat_cross_lay,
                                          cloud_scat_cross_int,
                                          g_0_tot_lay,
                                          g_0_tot_int,
                                          g_0,
                                          epsi,
                                          mu_star,
                                          scat,
                                          clouds);
        }

        cuda_check_status_or_exit(__FILE__, __LINE__);
        call_z_callback();

        direct_beam_flux(
            F_dir_wg, Fc_dir_wg, z_lay, mu_star, R_planet, R_star, a, dir_beam, geom_zenith_corr);

        cuda_check_status_or_exit(__FILE__, __LINE__);
    }

    int nscat_step = 0;
    if (single_walk)
        nscat_step = 200;
    else
        nscat_step = 3;

    for (int scat_iter = 0; scat_iter < nscat_step * scat + 1; scat_iter++) {
        if (iso) {
            populate_spectral_flux_iso(F_down_wg,   // out
                                       F_up_wg,     // out
                                       F_dir_wg,    // in
                                       g_0_tot_lay, // in
                                       g_0,
                                       single_walk,
                                       R_star,
                                       a,
                                       f_factor,
                                       mu_star,
                                       epsi,
                                       w_0_limit,
                                       dir_beam,
                                       clouds,
                                       albedo);
        }
        else {
            populate_spectral_flux_noniso(F_down_wg,
                                          F_up_wg,
                                          Fc_down_wg,
                                          Fc_up_wg,
                                          F_dir_wg,
                                          Fc_dir_wg,
                                          g_0_tot_lay,
                                          g_0_tot_int,
                                          g_0,
                                          single_walk,
                                          R_star,
                                          a,
                                          f_factor,
                                          mu_star,
                                          epsi,
                                          w_0_limit,
                                          delta_tau_limit,
                                          dir_beam,
                                          clouds,
                                          albedo,
                                          *trans_wg_upper,
                                          *trans_wg_lower);
        }

        cuda_check_status_or_exit(__FILE__, __LINE__);
    }


    double* gauss_weight = *gauss_weights;
    integrate_flux(deltalambda,
                   F_down_tot,
                   F_up_tot,
                   F_net,
                   F_down_wg,
                   F_up_wg,
                   F_dir_wg,
                   F_down_band,
                   F_up_band,
                   F_dir_band,
                   gauss_weight);


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
                  dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    double*       dev_T_int,           // in: it, pii, ioi, mmmi, kii
    double*       dev_p_lay,           // in: io, mmm, kil
    double*       dev_p_int,           // in: ioi, mmmi, kii
    double*       dev_opac_wg_lay,     // out: io
    double*       dev_opac_wg_int,     // out: ioi
    double*       dev_meanmolmass_lay, // out: mmm
    double*       dev_meanmolmass_int, // out: mmmi
    const bool&    real_star,           // pil
    const double& fake_opac,           // io
    const double& T_surf,              // csp, cse, pil
    const double& surf_albedo,         // cse
    const bool&   interpolate_temp_and_pres,
    const bool&   interp_and_calc_flux_step) {

    int nbin = opacities.nbin;


    // TODO: check where those planckband values are used, where used here in
    // calculate_surface_planck and correc_surface_emission that's not used anymore
    // out: csp, cse
    int plancktable_dim  = plancktable.dim;
    int plancktable_step = plancktable.step;

    if (interpolate_temp_and_pres) {
      // it
      dim3 it_grid(int((ninterface + 15) / 16), 1, 1);
      dim3 it_block(16, 1, 1);
      
      interpolate_temperature<<<it_grid, it_block>>>(dev_T_lay, // out
						     dev_T_int, // in
						     ninterface);
      cudaDeviceSynchronize();
    }
    
    // pil
    dim3 pil_grid(int((nbin + 15) / 16), int(((nlayer + 2) + 15)) / 16, 1);
    dim3 pil_block(16, 16, 1);
    planck_interpol_layer<<<pil_grid, pil_block>>>(dev_T_lay,                // in
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
        dim3 pii_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), 1);
        dim3 pii_block(16, 16, 1);
        planck_interpol_interface<<<pii_grid, pii_block>>>(dev_T_int,                // in
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
        dim3 io_grid(int((nbin + 15) / 16), int((nlayer + 15) / 16), 1);
        dim3 io_block(16, 16, 1);
        // TODO: should move fake_opac (opacity limit somewhere into opacity_table/interpolation component?)
        // out -> opacities (dev_opac_wg_lay)
        // out -> scetter cross section (scatter_cross_section_...)
        interpolate_opacities<<<io_grid, io_block>>>(dev_T_lay,                          // in
                                                     *opacities.dev_temperatures,        // in
                                                     dev_p_lay,                          // in
                                                     *opacities.dev_pressures,           // in
                                                     *opacities.dev_kpoints,             // in
                                                     dev_opac_wg_lay,                    // out
                                                     *opacities.dev_scat_cross_sections, // in
                                                     *scatter_cross_section_lay,         // out
                                                     opacities.n_pressures,
                                                     opacities.n_temperatures,
                                                     opacities.ny,
                                                     nbin,
                                                     fake_opac,
                                                     nlayer);


        cudaDeviceSynchronize();

        if (!iso) {
            // ioi
            dim3 ioi_grid(int((nbin + 15) / 16), int((ninterface + 15) / 16), 1);
            dim3 ioi_block(16, 16, 1);

            interpolate_opacities<<<ioi_grid, ioi_block>>>(dev_T_int,                   // in
                                                           *opacities.dev_temperatures, // in
                                                           dev_p_int,                   // in
                                                           *opacities.dev_pressures,    // in
                                                           *opacities.dev_kpoints,      // in
                                                           dev_opac_wg_int,             // out
                                                           *opacities.dev_scat_cross_sections, // in
                                                           *scatter_cross_section_inter, // out
                                                           opacities.n_pressures,
                                                           opacities.n_temperatures,
                                                           opacities.ny,
                                                           nbin,
                                                           fake_opac,
                                                           ninterface);

            cudaDeviceSynchronize();
        }

        // mmm
        dim3 mmm_block(16, 1, 1);
        dim3 mmm_grid(int((nlayer + 15) / 16), 1, 1);

        meanmolmass_interpol<<<mmm_grid, mmm_block>>>(dev_T_lay,                   // in
                                                      *opacities.dev_temperatures, // in
                                                      dev_meanmolmass_lay,         // out
                                                      *opacities.dev_meanmolmass,  // in
                                                      dev_p_lay,                   // in
                                                      *opacities.dev_pressures,    // in
                                                      opacities.n_pressures,
                                                      opacities.n_temperatures,
                                                      nlayer);


        cudaDeviceSynchronize();

        if (!iso) {
            // mmmi
            dim3 mmmi_block(16, 1, 1);
            dim3 mmmi_grid(int((ninterface + 15) / 16), 1, 1);

            meanmolmass_interpol<<<mmmi_grid, mmmi_block>>>(dev_T_int,                   // in
                                                            *opacities.dev_temperatures, // in
                                                            dev_meanmolmass_int,         // out
                                                            *opacities.dev_meanmolmass,  // in
                                                            dev_p_int,                   // in
                                                            *opacities.dev_pressures,    // in
                                                            opacities.n_pressures,
                                                            opacities.n_temperatures,
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
                                      double* F_net,
                                      double* F_down_wg,
                                      double* F_up_wg,
                                      double* F_dir_wg,
                                      double* F_down_band,
                                      double* F_up_band,
                                      double* F_dir_band,
                                      double* gauss_weight) {
  bool opt = true;
  
  int nbin = opacities.nbin;
  int ny   = opacities.ny;
  
  if (opt) {
    {
      int num_levels_per_block = 256/nbin + 1;
      dim3 gridsize(ninterface/num_levels_per_block + 1);
      dim3 blocksize(num_levels_per_block, nbin);
      //printf("nbin: %d, ny: %d\n", nbin, ny);
      
      integrate_flux_band<<<gridsize, blocksize>>>(F_down_wg,
						   F_up_wg,
						   F_dir_wg,
						   F_down_band,
						   F_up_band,
						   F_dir_band,
						   gauss_weight,
						   nbin,
						   ninterface,
						   ny);

      cudaDeviceSynchronize();
    }

    {
      int num_levels_per_block = 256/nbin + 1;
      dim3 gridsize(ninterface/num_levels_per_block + 1);
      dim3 blocksize(num_levels_per_block, nbin);
      integrate_flux_tot<<<gridsize, blocksize>>>(deltalambda,
						  F_down_tot,
						  F_up_tot,
						  F_net,
						  F_down_band,
						  F_up_band,
						  F_dir_band,
						  nbin,
						  ninterface);
      cudaDeviceSynchronize();
    }
    
    
  }
  else {
      
    dim3 threadsPerBlock(1, 1, 1);
    dim3 numBlocks(32, 4, 8);



    //printf("Running Alfrodull Wrapper for integrate flux\n");
    integrate_flux_double<<<threadsPerBlock, numBlocks>>>(deltalambda,
                                                          F_down_tot,
                                                          F_up_tot,
                                                          F_net,
                                                          F_down_wg,
                                                          F_up_wg,
                                                          F_dir_wg,
                                                          F_down_band,
                                                          F_up_band,
                                                          F_dir_band,
                                                          gauss_weight,
                                                          nbin,
                                                          ninterface,
                                                          ny);

    cudaDeviceSynchronize();
  }
}

bool alfrodull_engine::calculate_transmission_iso(double* trans_wg,             // out
                                                  double* delta_colmass,        // in
                                                  double* opac_wg_lay,          // in
                                                  double* cloud_opac_lay,       // in
                                                  double* meanmolmass_lay,      // in
                                                  double* cloud_scat_cross_lay, // in
                                                  double* g_0_tot_lay,          // in
                                                  double  g_0,
                                                  double  epsi,
                                                  double  mu_star,
                                                  bool    scat,
                                                  bool    clouds) {
    int nbin = opacities.nbin;

    int ny = opacities.ny;


    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
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
                               cloud_opac_lay,
                               meanmolmass_lay,
                               *scatter_cross_section_lay,
                               cloud_scat_cross_lay,
                               *w_0,
                               g_0_tot_lay,
                               g_0,
                               epsi,
                               mu_star,
                               w_0_limit,
                               scat,
                               nbin,
                               ny,
                               nlayer,
                               clouds,
                               scat_corr,
                               debug,
                               i2s_transition);

    cudaDeviceSynchronize();
    return true;
}

bool alfrodull_engine::calculate_transmission_noniso(double* trans_wg_upper,
                                                     double* trans_wg_lower,
                                                     double* delta_col_upper,
                                                     double* delta_col_lower,
                                                     double* opac_wg_lay,
                                                     double* opac_wg_int,
                                                     double* cloud_opac_lay,
                                                     double* cloud_opac_int,
                                                     double* meanmolmass_lay,
                                                     double* meanmolmass_int,
                                                     double* cloud_scat_cross_lay,
                                                     double* cloud_scat_cross_int,
                                                     double* g_0_tot_lay,
                                                     double* g_0_tot_int,
                                                     double  g_0,
                                                     double  epsi,
                                                     double  mu_star,
                                                     bool    scat,
                                                     bool    clouds) {
    int nbin = opacities.nbin;

    int ny = opacities.ny;

    dim3 grid(int((nbin + 15) / 16), int((ny + 3) / 4), int((nlayer + 3) / 4));
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
                                  cloud_opac_lay,
                                  cloud_opac_int,
                                  meanmolmass_lay,
                                  meanmolmass_int,
                                  *scatter_cross_section_lay,
                                  *scatter_cross_section_inter,
                                  cloud_scat_cross_lay,
                                  cloud_scat_cross_int,
                                  *w_0_upper,
                                  *w_0_lower,
                                  g_0_tot_lay,
                                  g_0_tot_int,
                                  g_0,
                                  epsi,
                                  mu_star,
                                  w_0_limit,
                                  scat,
                                  nbin,
                                  ny,
                                  nlayer,
                                  clouds,
                                  scat_corr,
                                  debug,
                                  i2s_transition);
    cudaDeviceSynchronize();
    return true;
}

bool alfrodull_engine::direct_beam_flux(double* F_dir_wg,
                                        double* Fc_dir_wg,
                                        double* z_lay,
                                        double  mu_star,
                                        double  R_planet,
                                        double  R_star,
                                        double  a,
                                        bool    dir_beam,
                                        bool    geom_zenith_corr) {

    int nbin = opacities.nbin;

    int ny = opacities.ny;


    if (iso) {
        dim3 block(4, 32, 4);
        dim3 grid(int((ninterface + 3) / 4), int((nbin + 31) / 32), int((ny + 3) / 4));
        fdir_iso<<<grid, block>>>(F_dir_wg,
                                  *planckband_lay,
                                  *delta_tau_wg,
                                  z_lay,
                                  mu_star,
                                  R_planet,
                                  R_star,
                                  a,
                                  dir_beam,
                                  geom_zenith_corr,
                                  ninterface,
                                  nbin,
                                  ny);

        cudaDeviceSynchronize();
    }
    else {
        dim3 block(4, 32, 4);
        dim3 grid(int((ninterface + 3) / 4), int((nbin + 31) / 32), int((ny + 3) / 4));

        fdir_noniso<<<grid, block>>>(F_dir_wg,
                                     Fc_dir_wg,
                                     *planckband_lay,
                                     *delta_tau_wg_upper,
                                     *delta_tau_wg_lower,
                                     z_lay,
                                     mu_star,
                                     R_planet,
                                     R_star,
                                     a,
                                     dir_beam,
                                     geom_zenith_corr,
                                     ninterface,
                                     nbin,
                                     ny);

        cudaDeviceSynchronize();
    }

    return true;
}

bool alfrodull_engine::populate_spectral_flux_iso(double* F_down_wg,   // out
                                                  double* F_up_wg,     // out
                                                  double* F_dir_wg,    // in
                                                  double* g_0_tot_lay, // in
                                                  double  g_0,
                                                  bool    singlewalk,
                                                  double  Rstar,
                                                  double  a,
                                                  double  f_factor,
                                                  double  mu_star,
                                                  double  epsi,
                                                  double  w_0_limit,
                                                  bool    dir_beam,
                                                  bool    clouds,
                                                  double  albedo) {

    int nbin = opacities.nbin;

    int ny = opacities.ny;


    dim3 block(16, 16, 1);
    dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);
    fband_iso_notabu<<<grid, block>>>(F_down_wg,
                                      F_up_wg,
                                      F_dir_wg,
                                      *planckband_lay,
                                      *w_0,
                                      *M_term,
                                      *N_term,
                                      *P_term,
                                      *G_plus,
                                      *G_minus,
                                      g_0_tot_lay,
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
                                      dir_beam,
                                      clouds,
                                      scat_corr,
                                      albedo,
                                      debug,
                                      i2s_transition);

    return true;
}

// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
bool alfrodull_engine::populate_spectral_flux_noniso(double* F_down_wg,
                                                     double* F_up_wg,
                                                     double* Fc_down_wg,
                                                     double* Fc_up_wg,
                                                     double* F_dir_wg,
                                                     double* Fc_dir_wg,
                                                     double* g_0_tot_lay,
                                                     double* g_0_tot_int,
                                                     double  g_0,
                                                     bool    singlewalk,
                                                     double  Rstar,
                                                     double  a,
                                                     double  f_factor,
                                                     double  mu_star,
                                                     double  epsi,
                                                     double  w_0_limit,
                                                     double  delta_tau_limit,
                                                     bool    dir_beam,
                                                     bool    clouds,
                                                     double  albedo,
                                                     double* trans_wg_upper,
                                                     double* trans_wg_lower) {
    int nbin = opacities.nbin;
    int ny   = opacities.ny;

    dim3 block(16, 16, 1);

    dim3 grid(int((nbin + 15) / 16), int((ny + 16) / 16), 1);

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

    return true;
}
