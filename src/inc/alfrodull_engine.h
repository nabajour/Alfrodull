#include "cudaDeviceMemory.h"
#include "opacities.h"
#include "planck_table.h"

#include <functional>

class alfrodull_engine
{
public:
    alfrodull_engine();

    void init();

    void load_opacities(const string& filename);
    void prepare_planck_table();

    void set_parameters(const int&    nlayer_,
                        const bool&   iso_,
                        const double& T_star_,
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
			const bool&   debug );

    bool   real_star   = false;
    double fake_opac   = false;
    double T_surf      = 0.0;
    double surf_albedo = 0.0;
    double g_0         = 0.0;
    double epsi        = 0.0;
    double mu_star     = 0.0;
    bool   scat        = false;
    bool   scat_corr        = false;
    double R_planet    = 0.0;
    double R_star      = 0.0;
    double a           = 0.0;
    bool   dir_beam    = false;

    bool geom_zenith_corr = false;

    double f_factor  = 0.0;
    double w_0_limit = 0.0;
    double albedo    = 0.0;

  
    double i2s_transition = 0.0;
    bool   debug = false;

    // call if using clouds, to set data array pointers
    void set_clouds_data(const bool& clouds,
                         double*     cloud_opac_lay,
                         double*     cloud_opac_int,
                         double*     cloud_scat_cross_lay,
                         double*     cloud_scat_cross_int,
                         double*     g_0_tot_lay,
                         double*     g_0_tot_int);

    double* cloud_opac_lay       = nullptr;
    double* cloud_opac_int       = nullptr;
    double* cloud_scat_cross_lay = nullptr;
    double* cloud_scat_cross_int = nullptr;
    double* g_0_tot_lay          = nullptr;
    double* g_0_tot_int          = nullptr;

    bool clouds = false;
    void allocate_internal_variables();

    // TODO: temporary prototyping wrapper for HELIOS.
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

    void correct_incident_energy(double* starflux_array_ptr,
                                 bool    real_star,
                                 bool    energy_budge_correction);

    void set_z_calc_func(std::function<void()>& fun);
    void call_z_callback();

    //private:
    opacity_table opacities;

    planck_table plancktable;

    // general sim parameters
    int nbin = 0; // should come from opacity table (?)

    int    nlayer     = 0;
    int    ninterface = 0; // nlayers + 1
    bool   iso        = false;
    double T_star     = 0.0;

    std::function<void()> calc_z_func;

    // device memory
    //  scattering
    cuda_device_memory<double> scatter_cross_section_lay;
    cuda_device_memory<double> scatter_cross_section_inter;

    // planck function
    cuda_device_memory<double> planckband_lay;
    cuda_device_memory<double> planckband_int;

    // delta tau, for weights. Only used internally (on device) for flux computations
    // and shared at the end for integration over wg
    // iso
    cuda_device_memory<double> delta_tau_wg;
    // noiso
    cuda_device_memory<double> delta_tau_wg_upper;
    cuda_device_memory<double> delta_tau_wg_lower;

    //
    cuda_device_memory<double> dev_T_int;
    cuda_device_memory<double> delta_col_mass;
    cuda_device_memory<double> delta_col_upper;
    cuda_device_memory<double> delta_col_lower;
    cuda_device_memory<double> meanmolmass_int;
    cuda_device_memory<double> meanmolmass_lay;
    cuda_device_memory<double> opac_wg_lay;
    cuda_device_memory<double> opac_wg_int;
    cuda_device_memory<double> trans_wg;
    cuda_device_memory<double> trans_wg_upper;
    cuda_device_memory<double> trans_wg_lower;

    // Flux computation quantities
    // computed in trans_iso/trans_noniso
    // used in populate_spectral_flux (iso/non_iso)
    // iso
    cuda_device_memory<double> M_term;
    cuda_device_memory<double> N_term;
    cuda_device_memory<double> P_term;
    cuda_device_memory<double> G_plus;
    cuda_device_memory<double> G_minus;
    cuda_device_memory<double> w_0;
    // noniso
    cuda_device_memory<double> M_upper;
    cuda_device_memory<double> M_lower;
    cuda_device_memory<double> N_upper;
    cuda_device_memory<double> N_lower;
    cuda_device_memory<double> P_upper;
    cuda_device_memory<double> P_lower;
    cuda_device_memory<double> G_plus_upper;
    cuda_device_memory<double> G_plus_lower;
    cuda_device_memory<double> G_minus_upper;
    cuda_device_memory<double> G_minus_lower;
    cuda_device_memory<double> w_0_upper;
    cuda_device_memory<double> w_0_lower;
};