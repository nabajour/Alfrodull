#pragma once

#include "cuda_device_memory.h"
#include "storage.h"
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
