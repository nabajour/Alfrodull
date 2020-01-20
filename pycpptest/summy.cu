
#include <cstdio>

extern "C" {
  void print_hello() {
  printf("Hello World!\n");
}

  int add_test(int a, int b)
  {
    return a + b;
  }
}


__global__ void summy(double *a, double *b, double *c, int s_x, int s_y) {
  int idx = threadIdx.x + threadIdx.y * s_x;
  c[idx] = a[idx] + b[idx];
}

extern "C" {
void call_summy(double *a, double *b, double *c, int s_x, int s_y) {
  dim3 threadsPerBlock(1, 1);
  dim3 numBlocks(s_x, s_y, 1);
  summy<<<numBlocks, threadsPerBlock>>>(a, b, c, s_x, s_y);

  cudaDeviceSynchronize();
}
}
