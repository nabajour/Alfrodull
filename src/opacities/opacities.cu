#include "opacities.h"

#include "physics_constants.h"
#include <cstdio>
#include <stdexcept>
#include <tuple>

template<class T>
std::tuple<std::unique_ptr<T[]>, int> read_table_to_host(storage& s, string table_name) {
    std::unique_ptr<T[]> data = nullptr;
    int                  size = 0;

    bool load_OK = s.read_table(table_name, data, size);

    if (!load_OK) {
        printf("Error reading key %s from table\n", table_name.c_str());
        throw std::runtime_error("error");
    }

    return std::make_tuple(std::move(data), size);
}

template<class T>
void push_table_to_device(std::unique_ptr<T[]>& data, int size, cuda_device_memory<T>& device_mem) {
    bool allocate_OK = device_mem.allocate(size);
    if (!allocate_OK) {
        printf("Error allocating device memory \n");
        throw std::runtime_error("error");
    }

    bool put_OK = device_mem.put(data);
    if (!put_OK) {
        printf("Error copying data from host to device\n");
        throw std::runtime_error("error");
    }
}

template<class T>
int read_table_to_device(storage&               s,
                         string                 table_name,
                         cuda_device_memory<T>& device_mem,
                         T                      scaling_factor = 1.0,
                         T                      scaling_offset = 0.0) {

    std::unique_ptr<T[]> data = nullptr;
    int                  size = 0;

    tie(data, size) = read_table_to_host<T>(s, table_name);
    if (scaling_factor != 1.0)
        for (int i = 0; i < size; i++) {
            data[i] *= scaling_factor;
            data[i] += scaling_offset;
        }
    push_table_to_device<T>(data, size, device_mem);

    return size;
}

opacity_table::opacity_table() {
}

bool opacity_table::load_opacity_table(const string& filename) {
#ifdef CGS_UNITS
    // #warning "Compiling with CGS units"
    const double temperatures_unit_conv = 1.0;
    const double pressures_unit_conv    = 1.0;
    const double wavelength_unit_conv   = 1.0;
    const double opacity_unit_conv      = 1.0;
    const double scat_cross_unit_conv   = 1.0;
#else // SI units
    // #warning "Compiling with SI units"
    // for SI units, convert CGS to SI
    const double temperatures_unit_conv = 1.0;
    const double pressures_unit_conv    = 1.0e-1;
    const double wavelength_unit_conv   = 1.0e-2;
    const double opacity_unit_conv      = 1.0e-1;
    const double scat_cross_unit_conv   = 1.0e-4;

#endif // CGS_UNIT


    printf("Loading tables\n");
    storage s(filename, true);

    //read_table_to_device<double>(s, "/kpoints", dev_kpoints, opacity_unit_conv, 0.00125);
    //read_table_to_device<double>(s, "/kpoints", dev_kpoints, opacity_unit_conv, 0.025);
    //read_table_to_device<double>(s, "/kpoints", dev_kpoints, opacity_unit_conv, 0.0025);
    read_table_to_device<double>(
        s, "/kpoints", dev_kpoints, opacity_unit_conv, experimental_opacities_offset);
    n_temperatures =
        read_table_to_device<double>(s, "/temperatures", dev_temperatures, temperatures_unit_conv);
    n_pressures = read_table_to_device<double>(s, "/pressures", dev_pressures, pressures_unit_conv);
    read_table_to_device<double>(
        s, "/weighted Rayleigh cross-sections", dev_scat_cross_sections, scat_cross_unit_conv);
    {
        std::unique_ptr<double[]> data = nullptr;
        int                       size = 0;

        tie(data, size) = read_table_to_host<double>(s, "/meanmolmass");
        for (int i = 0; i < size; i++)
            data[i] *= AMU;
        push_table_to_device<double>(data, size, dev_meanmolmass);
    }

    std::unique_ptr<double[]> data_opac_wave = nullptr;

    if (s.has_table("/center wavelengths")) {
        tie(data_opac_wave, nbin) = read_table_to_host<double>(s, "/center wavelengths");
    }
    else {
        tie(data_opac_wave, nbin) = read_table_to_host<double>(s, "/wavelengths");
    }
    for (int i = 0; i < nbin; i++)
        data_opac_wave[i] *= wavelength_unit_conv;
    push_table_to_device<double>(data_opac_wave, nbin, dev_opac_wave);

    if (s.has_table("/ypoints")) {
        ny = read_table_to_device<double>(s, "/ypoints", dev_opac_y);
    }
    else {
        std::unique_ptr<double[]> data(new double[1]);
        data[0]  = 0;
        int size = 1;
        push_table_to_device<double>(data, size, dev_opac_y);
        ny = 1;
    }


    std::unique_ptr<double[]> data_opac_interwave(new double[nbin + 1]);
    if (s.has_table("/interface wavelengths")) {
        read_table_to_device<double>(
            s, "/interface wavelengths", dev_opac_interwave, wavelength_unit_conv);
    }
    else {
        // TODO : check those interpolated values usage
        // TODO: dump tables to compare wiwth original
        // quick and dirty way to get the lamda interface values
        data_opac_interwave[0] = data_opac_wave[0] - (data_opac_wave[1] - data_opac_wave[0]) / 2.0;
        for (int i = 0; i < nbin - 1; i++)
            data_opac_interwave[i + 1] = (data_opac_wave[i + 1] + data_opac_wave[i]) / 2.0;
        data_opac_interwave[nbin] =
            data_opac_wave[nbin - 1] + (data_opac_wave[nbin - 1] - data_opac_wave[nbin - 2]) / 2.0;

        // for (int i = 0; i < nbin; i++)
        //   data_opac_interwave[i] *= wavelength_unit_conv;
        push_table_to_device<double>(data_opac_interwave, nbin + 1, dev_opac_interwave);
    }
    // for (int i = 0; i < nbin + 1; i++)
    //   printf("interwave %d %g\n", i, data_opac_interwave[i]);

    if (s.has_table("/wavelength width of bins")) {
        read_table_to_device<double>(
            s, "/wavelength width of bins", dev_opac_deltawave, wavelength_unit_conv);
    }
    else {
        // TODO : check those interpolated values usage
        if (nbin == 1) {
            std::unique_ptr<double[]> data_opac_deltawave(new double[1]);
            data_opac_deltawave[0] = 0.0;
            push_table_to_device<double>(data_opac_deltawave, 1, dev_opac_deltawave);
        }
        else {

            std::unique_ptr<double[]> data_opac_deltawave(new double[nbin]);
            // unit already converted with interwave
            for (int i = 0; i < nbin; i++)
                data_opac_deltawave[i] = data_opac_interwave[i + 1] - data_opac_interwave[i];
            push_table_to_device<double>(data_opac_deltawave, nbin, dev_opac_deltawave);
            // for (int i = 0; i < nbin; i++)
            //   printf("deltawave %d %g\n", i, data_opac_deltawave[i]);
        }
    }


    return true;
}
