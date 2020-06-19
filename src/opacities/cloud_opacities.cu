#include "cloud_opacities.h"

#include "physics_constants.h"
#include <cstdio>
#include <stdexcept>
#include <tuple>

#include "opacities_helpers.h"

cloud_opacity_table::cloud_opacity_table() {
}

bool cloud_opacity_table::load(const string& filename) {

    printf("Loading tables\n");
    storage s(filename, true);

    // TODO add loading of wavelength and ypoints
    // TODO outside of here, check that it matches opacities bins

    read_table_to_device<double>(s, "/asymmetry", dev_asymmetry);

    read_table_to_device<double>(s, "/scattering", dev_scat_cross_sections);

    read_table_to_device<double>(s, "/absorption", dev_abs_cross_sections);

    cudaError_t err = cudaGetLastError();

    // Check device query
    if (err != cudaSuccess) {
        log::printf("[%s:%d] CUDA error check reports error: %s\n",
                    __FILE__,
                    __LINE__,
                    cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return true;
}
