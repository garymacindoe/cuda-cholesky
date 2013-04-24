#ifndef __CAXPY_H
#define __CAXPY_H
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:8) += alpha * x(1:8)
__device__ void caxpy(cuComplex alpha, const float * x_real, const float * x_imag, cuComplex * y) {
  y[0] = cuCfmaf(alpha, make_cuComplex(x_real[0], x_imag[0]), y[0]);
  y[1] = cuCfmaf(alpha, make_cuComplex(x_real[1], x_imag[1]), y[1]);
  y[2] = cuCfmaf(alpha, make_cuComplex(x_real[2], x_imag[2]), y[2]);
  y[3] = cuCfmaf(alpha, make_cuComplex(x_real[3], x_imag[3]), y[3]);
  y[4] = cuCfmaf(alpha, make_cuComplex(x_real[4], x_imag[4]), y[4]);
  y[5] = cuCfmaf(alpha, make_cuComplex(x_real[5], x_imag[5]), y[5]);
  y[6] = cuCfmaf(alpha, make_cuComplex(x_real[6], x_imag[6]), y[6]);
  y[7] = cuCfmaf(alpha, make_cuComplex(x_real[7], x_imag[7]), y[7]);
}

// y(1:8) += x(1:8)
__device__ void caxpy(const float * __restrict__ x_real,
                      const float * __restrict__ x_imag, cuComplex * __restrict__ y) {
  y[0] = cuCaddf(y[0], make_cuComplex(x_real[0], x_imag[0]));
  y[1] = cuCaddf(y[1], make_cuComplex(x_real[1], x_imag[1]));
  y[2] = cuCaddf(y[2], make_cuComplex(x_real[2], x_imag[2]));
  y[3] = cuCaddf(y[3], make_cuComplex(x_real[3], x_imag[3]));
  y[4] = cuCaddf(y[4], make_cuComplex(x_real[4], x_imag[4]));
  y[5] = cuCaddf(y[5], make_cuComplex(x_real[5], x_imag[5]));
  y[6] = cuCaddf(y[6], make_cuComplex(x_real[6], x_imag[6]));
  y[7] = cuCaddf(y[7], make_cuComplex(x_real[7], x_imag[7]));
}

// y(1:n) += alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const float * __restrict__ x_real,
                      const float * __restrict__ x_imag, cuComplex * __restrict__ y) {
  if (n <= 0) return;
  y[0] = cuCfmaf(alpha, make_cuComplex(x_real[0], x_imag[0]), y[0]); if (1 >= n) return;
  y[1] = cuCfmaf(alpha, make_cuComplex(x_real[1], x_imag[1]), y[1]); if (2 >= n) return;
  y[2] = cuCfmaf(alpha, make_cuComplex(x_real[2], x_imag[2]), y[2]); if (3 >= n) return;
  y[3] = cuCfmaf(alpha, make_cuComplex(x_real[3], x_imag[3]), y[3]); if (4 >= n) return;
  y[4] = cuCfmaf(alpha, make_cuComplex(x_real[4], x_imag[4]), y[4]); if (5 >= n) return;
  y[5] = cuCfmaf(alpha, make_cuComplex(x_real[5], x_imag[5]), y[5]); if (6 >= n) return;
  y[6] = cuCfmaf(alpha, make_cuComplex(x_real[6], x_imag[6]), y[6]); if (7 >= n) return;
  y[7] = cuCfmaf(alpha, make_cuComplex(x_real[7], x_imag[7]), y[7]);
}

#else

// y(1:8) += alpha * x(1:8)
__device__ void caxpy(cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCfmaf(alpha, x[0], y[0]); y[1] = cuCfmaf(alpha, x[1], y[1]);
  y[2] = cuCfmaf(alpha, x[2], y[2]); y[3] = cuCfmaf(alpha, x[3], y[3]);
  y[4] = cuCfmaf(alpha, x[4], y[4]); y[5] = cuCfmaf(alpha, x[5], y[5]);
  y[6] = cuCfmaf(alpha, x[6], y[6]); y[7] = cuCfmaf(alpha, x[7], y[7]);
}

// y(1:8) += x(1:8)
__device__ void caxpy(const cuComplex * x, cuComplex * y) {
  y[0] = cuCaddf(y[0], x[0]); y[1] = cuCaddf(y[1], x[1]);
  y[2] = cuCaddf(y[2], x[2]); y[3] = cuCaddf(y[3], x[3]);
  y[4] = cuCaddf(y[4], x[4]); y[5] = cuCaddf(y[5], x[5]);
  y[6] = cuCaddf(y[6], x[6]); y[7] = cuCaddf(y[7], x[7]);
}

// y(1:n) += alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const cuComplex * x, cuComplex * y) {
  if (n <= 0) return;
  y[0] = cuCfmaf(alpha, x[0], y[0]); if (1 >= n) return; y[1] = cuCfmaf(alpha, x[1], y[1]); if (2 >= n) return;
  y[2] = cuCfmaf(alpha, x[2], y[2]); if (3 >= n) return; y[3] = cuCfmaf(alpha, x[3], y[3]); if (4 >= n) return;
  y[4] = cuCfmaf(alpha, x[4], y[4]); if (5 >= n) return; y[5] = cuCfmaf(alpha, x[5], y[5]); if (6 >= n) return;
  y[6] = cuCfmaf(alpha, x[6], y[6]); if (7 >= n) return; y[7] = cuCfmaf(alpha, x[7], y[7]);
}

#endif

#endif
