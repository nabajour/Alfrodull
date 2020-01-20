import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule

a = np.random.randn(4, 4).astype(np.float64)
b = np.random.randn(4, 4).astype(np.float64)
c = np.zeros((4, 4), dtype=np.float64)

mod = SourceModule("""
  __global__ void summy(double *a, double *b, double *c, int s_x, int s_y)
  {
    int idx = threadIdx.x + threadIdx.y*s_x;
    c[idx] = a[idx] + b[idx];
  }
  """)


func = mod.get_function("summy")
func(drv.In(a), drv.In(b), drv.Out(c), 4, 4, block=(4, 4, 1), grid=(1, 1))

print(c)
