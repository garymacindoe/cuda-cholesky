#ifndef __ZAXPY_H
#define __ZAXPY_H
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy4(cuDoubleComplex alpha, const int * __restrict__ x_real_hi, const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi, const int * __restrict__ x_imag_lo, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[0], x_real_lo[0]),
                     __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);

  y[1] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[1], x_real_lo[1]),
                     __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);

  y[2] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[2], x_real_lo[2]),
                     __hiloint2double(x_imag_hi[2], x_imag_lo[2])), y[2]);

  y[3] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[3], x_real_lo[3]),
                     __hiloint2double(x_imag_hi[3], x_imag_lo[3])), y[3]);
}

// y(1:2) += alpha * x(1:2)
__device__ void zaxpy2(cuDoubleComplex alpha, const int * __restrict__ x_real_hi, const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi, const int * __restrict__ x_imag_lo, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[0], x_real_lo[0]),
                     __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);

  y[1] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[1], x_real_lo[1]),
                     __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);
}

// y(1:4) += x(1:4)
__device__ void zaxpy4(const int * __restrict__ x_real_hi,
                       const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi,
                       const int * __restrict__ x_imag_lo,
                       cuDoubleComplex * __restrict__ y) {
  y[0] = cuCadd(y[0], make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                           __hiloint2double(x_imag_hi[0], x_imag_lo[0])));
  y[1] = cuCadd(y[1], make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                           __hiloint2double(x_imag_hi[1], x_imag_lo[1])));
  y[2] = cuCadd(y[2], make_cuDoubleComplex(__hiloint2double(x_real_hi[2], x_real_lo[2]),
                                           __hiloint2double(x_imag_hi[2], x_imag_lo[2])));
  y[3] = cuCadd(y[3], make_cuDoubleComplex(__hiloint2double(x_real_hi[3], x_real_lo[3]),
                                           __hiloint2double(x_imag_hi[3], x_imag_lo[3])));
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy4(int n, cuDoubleComplex alpha,
                       const int * __restrict__ x_real_hi,
                       const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi,
                       const int * __restrict__ x_imag_lo,
                       cuDoubleComplex * __restrict__ y) {
  if (n <= 0) return;
  y[0] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                            __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);
  if (1 >= n) return;
  y[1] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                            __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);
  if (2 >= n) return;
  y[2] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[2], x_real_lo[2]),
                                            __hiloint2double(x_imag_hi[2], x_imag_lo[2])), y[2]);
  if (3 >= n) return;
  y[3] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[3], x_real_lo[3]),
                                            __hiloint2double(x_imag_hi[3], x_imag_lo[3])), y[3]);
}

#else

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy4(cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ x, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, x[0], y[0]); y[1] = cuCfma(alpha, x[1], y[1]);
  y[2] = cuCfma(alpha, x[2], y[2]); y[3] = cuCfma(alpha, x[3], y[3]);
}

// y(1:2) += alpha * x(1:2)
__device__ void zaxpy2(cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ x, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, x[0], y[0]); y[1] = cuCfma(alpha, x[1], y[1]);
}

// y(1:4) += x(1:4)
__device__ void zaxpy4(const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCadd(y[0], x[0]); y[1] = cuCadd(y[1], x[1]);
  y[2] = cuCadd(y[2], x[2]); y[3] = cuCadd(y[3], x[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy4(int n, cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  if (n <= 0) return;
  y[0] = cuCfma(alpha, x[0], y[0]); if (1 >= n) return; y[1] = cuCfma(alpha, x[1], y[1]); if (2 >= n) return;
  y[2] = cuCfma(alpha, x[2], y[2]); if (3 >= n) return; y[3] = cuCfma(alpha, x[3], y[3]);
}

#endif

#endif
