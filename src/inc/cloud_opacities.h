#pragma once

#include <memory>
#include <string>


#include "cuda_device_memory.h"
#include "storage.h"

// class to handle cloud scattering cross scection, absorption cross section and asymmetry
// loads data table
// initialise variables for ref table management
// prepares GPU data
// loads data to GPU
// interpolates data
class cloud_opacity_table
{
public:
    cloud_opacity_table();

    bool load(const string& filename);

    // device tables
    // scattering cross section
    cuda_device_memory<double> dev_scat_cross_sections;

    // absorption cross section
    cuda_device_memory<double> dev_abs_cross_sections;

    // asymmetry
    cuda_device_memory<double> dev_asymmetry;
};
