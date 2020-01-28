
// class to handle opacities and meanmolmass from k_table
// loads data table
// initialise variables for ref table management
// prepares GPU data
// loads data to GPU
// interpolates data
class OpacityTable
{
public:
    OpacityTable();


private:
    string opacity_filename;

    bool load_opacity_table(const string& filename);
    bool copy_opacity_table_to_device();


    // TODO: read directly and push to device ?
    std::unique_ptr<H5File> file;

    // HDF5 tables
    std::unique_ptr<double> opac_kpoints;

    std::unique_ptr<double> opac_temperatures;
    int                     n_temp = 0;
    std::unique_ptr<double> opac_pressures;
    int                     n_pressures = 0;

    // wieghted Rayeigh c
    std::unique_ptr<double> scat_cross_sections;

    // Mean molecular mass
    // TODO: needs to be in AMU
    std::unique_ptr<double> mean_mol_mass;

    std::unique_ptr<double> opac_wave;
    int                     nbin = 0;

    std::unique_ptr<double> opac_y;
    int                     ny;

    std::unique_ptr<double> opac_interwave;

    std::unique_ptr<double> opac_deltawave;

    // needed for interpolate_opacities
    // dev_T_lay
    // dev_ktemp
    // dev_p_lay,
    // 					      dev_kpress,
    // 						    dev_opac_k,
    // 						    dev_opac_wg_lay,
    // 						    dev_opac_scat_cross,
    // 						    dev_scat_cross_lay,
    // 						    npress,
    // 						    ntemp,
    // 						    ny,
    // 						    nbin,
    // 						    fake_opac,
    // 						    nlayer
};
