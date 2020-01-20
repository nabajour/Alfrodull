
import numpy as np

# CUDA
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# CFFI
from cffi import FFI
import ctypes

# CUDA test
a = np.ones((4, 4), dtype=np.float64)
b = np.ones((4, 4), dtype=np.float64)*2

c = np.zeros((4, 4), dtype=np.float64)

mod = SourceModule("""
  __global__ void summy(double *a, double *b, double *c, int s_x, int s_y)
  {
    int idx = threadIdx.x + threadIdx.y*s_x;
    c[idx] = a[idx] + b[idx];
  }
  """)


#func = mod.get_function("summy")
# func(drv.In(a), drv.In(b), drv.Out(c), np.int32(
#    4), np.int32(4), block=(4, 4, 1), grid=(1, 1))
#
# print(c)

# CFFI test
ffibuilder = FFI()
ffibuilder.cdef("""
                   void print_hello();

                   int add_test(int a, int b);

                   void call_summy(double *a, double *b, double *c, int s_x, int s_y);

""")
ffibuilder.set_source("_summy",
                      """
               # include "summy.h"
               """, libraries=["summy"], library_dirs=["./"])

ffibuilder.compile(verbose=True)

print("Import wrapped lib")

from _summy import ffi, lib  # noqa

print("printing test")
lib.print_hello()

aplusb = lib.add_test(2, 4)
print(f"a + b = {aplusb}")

print("CUDA test")

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.to_gpu(c)

lib.call_summy(ffi.cast("double *", a_gpu.ptr),
               ffi.cast("double *", b_gpu.ptr),
               ffi.cast("double *", c_gpu.ptr),
               4, 4)
c_res = c_gpu.get()
print(c_res)
