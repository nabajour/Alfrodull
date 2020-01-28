#include "opacities.h"

#include <cstdio>

template<class T>
void read_table_to_device(storage& s, string table_name, cuda_device_memory<T>& device_mem) {

    std::unique_ptr<T[]> data    = nullptr;
    int                  size    = 0;
    bool                 load_OK = s.read_table(table_name, data, size);

    if (!load_OK) {
        printf("Error reading key %s from table\n", table_name.c_str());
        return;
    }

    bool allocate_OK = device_mem.allocate(size);
    if (!allocate_OK) {
        printf("Error allocating decice memory for table %s\n", table_name.c_str());
        return;
    }

    bool put_OK = device_mem.put(data);
    if (!put_OK) {
        printf("Error copying data from host to device for table %s\n", table_name.c_str());
        return;
    }
}

opacity_table::opacity_table() {
}

bool opacity_table::load_opacity_table(const string& filename) {
    printf("Loading tables\n");
    storage s(filename, true);

    read_table_to_device<double>(s, "/kpoints", dev_kpoints);

    return true;
}
