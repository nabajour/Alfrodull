#include "opacities.h"

#include <cstdio>

template<class T>
void read_table_to_device(storage & s, string table_name, cuda_device_memory<T> & device_mem)
{
  
  std::unique_ptr<T[]> data = nullptr;
  int size = 0;
  bool load_OK = s.read_table(table_name, data, size);

  if (!load_OK) {
    printf("Error reading key %s from table\n", table_name.c_str());
    return;
  }
  device_mem.allocate(size);

  device_mem.put(data);
}

opacity_table::opacity_table()
{

}

bool opacity_table::load_opacity_table(const string & filename)
{
  storage s(filename, true);

  read_table_to_device<double>(s, "/kpoints", dev_kpoints);

  return true;
}
