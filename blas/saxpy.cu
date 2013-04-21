#ifndef __SAXPY_H
#define __SAXPY_H

// y(1:16) += alpha * x(1:16)
__device__ void saxpy(float alpha, const float * __restrict__ x, float * __restrict__ y) {
  y[ 0] += alpha * x[ 0]; y[ 1] += alpha * x[ 1]; y[ 2] += alpha * x[ 2]; y[ 3] += alpha * x[ 3];
  y[ 4] += alpha * x[ 4]; y[ 5] += alpha * x[ 5]; y[ 6] += alpha * x[ 6]; y[ 7] += alpha * x[ 7];
  y[ 8] += alpha * x[ 8]; y[ 9] += alpha * x[ 9]; y[10] += alpha * x[10]; y[11] += alpha * x[11];
  y[12] += alpha * x[12]; y[13] += alpha * x[13]; y[14] += alpha * x[14]; y[15] += alpha * x[15];
}

// y(1:16) += x(1:16)
__device__ void saxpy(const float * __restrict__ x, float * __restrict__ y) {
  y[ 0] += x[ 0]; y[ 1] += x[ 1]; y[ 2] += x[ 2]; y[ 3] += x[ 3];
  y[ 4] += x[ 4]; y[ 5] += x[ 5]; y[ 6] += x[ 6]; y[ 7] += x[ 7];
  y[ 8] += x[ 8]; y[ 9] += x[ 9]; y[10] += x[10]; y[11] += x[11];
  y[12] += x[12]; y[13] += x[13]; y[14] += x[14]; y[15] += x[15];
}

// y(1:n) += alpha * x(1:n)
__device__ void saxpy(int n, float alpha, const float * __restrict__ x, float * __restrict__ y) {
  if (n <= 0) return;
  y[ 0] += alpha * x[ 0]; if ( 1 >= n) return; y[ 1] += alpha * x[ 1]; if ( 2 >= n) return;
  y[ 2] += alpha * x[ 2]; if ( 3 >= n) return; y[ 3] += alpha * x[ 3]; if ( 4 >= n) return;
  y[ 4] += alpha * x[ 4]; if ( 5 >= n) return; y[ 5] += alpha * x[ 5]; if ( 6 >= n) return;
  y[ 6] += alpha * x[ 6]; if ( 7 >= n) return; y[ 7] += alpha * x[ 7]; if ( 8 >= n) return;
  y[ 8] += alpha * x[ 8]; if ( 9 >= n) return; y[ 9] += alpha * x[ 9]; if (10 >= n) return;
  y[10] += alpha * x[10]; if (11 >= n) return; y[11] += alpha * x[11]; if (12 >= n) return;
  y[12] += alpha * x[12]; if (13 >= n) return; y[13] += alpha * x[13]; if (14 >= n) return;
  y[14] += alpha * x[14]; if (15 >= n) return; y[15] += alpha * x[15];
}

#endif
