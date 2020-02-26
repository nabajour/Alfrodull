#include <functional>
#include <tuple>

void wrap_compute_radiative_transfer(
    // prepare_compute_flux
    long dev_starflux, // in: pil
    // state variables
    long
                dev_T_lay, // out: it, pil, io, mmm, kil   (interpolated from T_int and then used as input to other funcs)
    long        dev_T_int, // in: it, pii, ioi, mmmi, kii
    long        dev_p_lay, // in: io, mmm, kil
    long        dev_p_int, // in: ioi, mmmi, kii
    const bool& interp_and_calc_flux_step,
    // direct_beam_flux
    long z_lay,
    // spectral flux loop
    bool single_walk,
    // populate_spectral_flux_noniso
    long   F_down_wg,
    long   F_up_wg,
    long   Fc_down_wg,
    long   Fc_up_wg,
    long   F_dir_wg,
    long   Fc_dir_wg,
    double delta_tau_limit,
    // integrate_flux
    long F_down_tot,
    long F_up_tot,
    long F_net,
    long F_down_band,
    long F_up_band,
    long F_dir_band);

void init_alfrodull();
void init_parameters(const int&    nlayer_,
                     const bool&   iso_,
                     const double& Tstar_,
                     const bool&   real_star,
                     const double& fake_opac,
                     const double& T_surf,
                     const double& surf_albedo,
                     const double& g_0,
                     const double& epsi,
                     const double& mu_star,
                     const bool&   scat,
                     const bool&   scat_corr,
                     const double& R_planet,
                     const double& R_star,
                     const double& a,
                     const bool&   dir_beam,
                     const bool&   geom_zenith_corr,
                     const double& f_factor,
                     const double& w_0_limit,
                     const double& albedo,
                     const double& i2s_transition,
                     const bool&   debug);

void set_surface_temperature(const double& T_surf);
void deinit_alfrodull();

void set_clouds_data(const bool& clouds_,
                     const long& cloud_opac_lay_,
                     const long& cloud_opac_int_,
                     const long& cloud_scat_cross_lay_,
                     const long& cloud_scat_cross_int_,
                     const long& g_0_tot_lay_,
                     const long& g_0_tot_int_);

void set_z_calc_function(std::function<void()>& func);

// TODO: this shouldn't be visible externally
void allocate();

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
get_device_pointers_for_helios_write();

std::tuple<long, long, long, long, int, int> get_opac_data_for_helios();

void prepare_planck_table();
void correct_incident_energy(long starflux_array_ptr, bool real_star, bool energy_budge_correction);
