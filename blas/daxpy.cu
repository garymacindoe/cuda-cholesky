#ifndef __DAXPY_H
#define __DAXPY_H

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:8) += alpha * x(1:8)
__device__ void daxpy(double alpha, const int * __restrict__ x_hi,
                      const int * __restrict__ x_lo, double * __restrict__ y) {
  y[0] += alpha * __hiloint2double(x_hi[0], x_lo[0]);
  y[1] += alpha * __hiloint2double(x_hi[1], x_lo[1]);
  y[2] += alpha * __hiloint2double(x_hi[2], x_lo[2]);
  y[3] += alpha * __hiloint2double(x_hi[3], x_lo[3]);
  y[4] += alpha * __hiloint2double(x_hi[4], x_lo[4]);
  y[5] += alpha * __hiloint2double(x_hi[5], x_lo[5]);
  y[6] += alpha * __hiloint2double(x_hi[6], x_lo[6]);
  y[7] += alpha * __hiloint2double(x_hi[7], x_lo[7]);
}

// y(1:8) += x(1:8)
__device__ void daxpy(const int * __restrict__ x_hi,
                      const int * __restrict__ x_lo, double * __restrict__ y) {
  y[0] += __hiloint2double(x_hi[0], x_lo[0]);
  y[1] += __hiloint2double(x_hi[1], x_lo[1]);
  y[2] += __hiloint2double(x_hi[2], x_lo[2]);
  y[3] += __hiloint2double(x_hi[3], x_lo[3]);
  y[4] += __hiloint2double(x_hi[4], x_lo[4]);
  y[5] += __hiloint2double(x_hi[5], x_lo[5]);
  y[6] += __hiloint2double(x_hi[6], x_lo[6]);
  y[7] += __hiloint2double(x_hi[7], x_lo[7]);
}

// y(1:n) += alpha * x(1:n)
__device__ void daxpy(int n, double alpha, const int * __restrict__ x_hi,
                      const int * __restrict__ x_lo, double * __restrict__ y) {
  if (n <= 0) return;
  y[0] += alpha * __hiloint2double(x_hi[0], x_lo[0]); if (1 >= n) return;
  y[1] += alpha * __hiloint2double(x_hi[1], x_lo[1]); if (2 >= n) return;
  y[2] += alpha * __hiloint2double(x_hi[2], x_lo[2]); if (3 >= n) return;
  y[3] += alpha * __hiloint2double(x_hi[3], x_lo[3]); if (4 >= n) return;
  y[4] += alpha * __hiloint2double(x_hi[4], x_lo[4]); if (5 >= n) return;
  y[5] += alpha * __hiloint2double(x_hi[5], x_lo[5]); if (6 >= n) return;
  y[6] += alpha * __hiloint2double(x_hi[6], x_lo[6]); if (7 >= n) return;
  y[7] += alpha * __hiloint2double(x_hi[7], x_lo[7]);
}

#else

// y(1:8) += alpha * x(1:8)
__device__ void daxpy(double alpha, const double * __restrict__ x, double * __restrict__ y) {
  y[0] += alpha * x[0]; y[1] += alpha * x[1]; y[2] += alpha * x[2]; y[3] += alpha * x[3];
  y[4] += alpha * x[4]; y[5] += alpha * x[5]; y[6] += alpha * x[6]; y[7] += alpha * x[7];
}

// y(1:8) += x(1:8)
__device__ void daxpy(const double * x, double * y) {
  y[ 0] += x[ 0]; y[ 1] += x[ 1]; y[ 2] += x[ 2]; y[ 3] += x[ 3];
  y[ 4] += x[ 4]; y[ 5] += x[ 5]; y[ 6] += x[ 6]; y[ 7] += x[ 7];
}

// y(1:n) += alpha * x(1:n)
__device__ void daxpy(int n, double alpha, const double * x, double * y) {
  if (n <= 0) return;
  y[ 0] += alpha * x[ 0]; if ( 1 >= n) return; y[ 1] += alpha * x[ 1]; if ( 2 >= n) return;
  y[ 2] += alpha * x[ 2]; if ( 3 >= n) return; y[ 3] += alpha * x[ 3]; if ( 4 >= n) return;
  y[ 4] += alpha * x[ 4]; if ( 5 >= n) return; y[ 5] += alpha * x[ 5]; if ( 6 >= n) return;
  y[ 6] += alpha * x[ 6]; if ( 7 >= n) return; y[ 7] += alpha * x[ 7];
}

#endif

#endif
