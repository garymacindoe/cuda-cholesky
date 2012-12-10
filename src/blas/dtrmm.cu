#include "blas.h"

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICT__)

// y(1:8) += alpha * x(1:8)
__device__ void daxpy(double alpha, const int * x_hi, const int * x_lo, double * y) {
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
__device__ void daxpy(const int * x_hi, const int * x_lo, double * y) {
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
__device__ void daxpy(int n, double alpha, const int * x_hi, const int * x_lo, double * y) {
  y[0] += alpha * __hiloint2double(x_hi[0], x_lo[0]); if ( 1 >= n) return;
  y[1] += alpha * __hiloint2double(x_hi[1], x_lo[1]); if ( 2 >= n) return;
  y[2] += alpha * __hiloint2double(x_hi[2], x_lo[2]); if ( 3 >= n) return;
  y[3] += alpha * __hiloint2double(x_hi[3], x_lo[3]); if ( 4 >= n) return;
  y[4] += alpha * __hiloint2double(x_hi[4], x_lo[4]); if ( 5 >= n) return;
  y[5] += alpha * __hiloint2double(x_hi[5], x_lo[5]); if ( 6 >= n) return;
  y[6] += alpha * __hiloint2double(x_hi[6], x_lo[6]); if ( 7 >= n) return;
  y[7] += alpha * __hiloint2double(x_hi[7], x_lo[7]);
}

// y(1:n) = alpha * x(1:n)
__device__ void dscal(int n, double alpha, const double * x, double * y, int incy) {
  if (n <= 0) return;
  y[0] = alpha * x[ 0]; if ( 1 >= n) return; y += incy;
  y[0] = alpha * x[ 1]; if ( 2 >= n) return; y += incy;
  y[0] = alpha * x[ 2]; if ( 3 >= n) return; y += incy;
  y[0] = alpha * x[ 3]; if ( 4 >= n) return; y += incy;
  y[0] = alpha * x[ 4]; if ( 5 >= n) return; y += incy;
  y[0] = alpha * x[ 5]; if ( 6 >= n) return; y += incy;
  y[0] = alpha * x[ 6]; if ( 7 >= n) return; y += incy;
  y[0] = alpha * x[ 7];
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ int b_hi[kb][nb];
  __shared__ int b_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          daxpy(A[0], b_hi[ll], b_lo[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(b_hi[ll], b_lo[ll], x);
        else if (ti < l)
          daxpy(A[0], b_hi[ll], b_lo[ll], x);
        A += lda;
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti <= l++)
        daxpy(A[0], b_hi[ll], b_lo[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(b_hi[ll], b_lo[ll], x);
      else if (ti < l)
        daxpy(A[0], b_hi[ll], b_lo[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for DGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(A[0], b_hi[l], b_lo[l], x);
      A += lda;
    }

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(A[0], b_hi[l], b_lo[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ int a_hi[mb][kb + 1];
  __shared__ int a_lo[mb][kb + 1];
  __shared__ int b_hi[kb][nb];
  __shared__ int b_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(A[i * lda]);
      a_lo[i + threadIdx.y][threadIdx.x] = __double2loint(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      daxpy(__hiloint2double(a_hi[ti][l], a_lo[ti][l]),
            &b_hi[l][tj], &b_lo[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(A[i * lda]);
      a_lo[i + threadIdx.y][threadIdx.x] = __double2loint(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
                &b_hi[ll][tj], &b_lo[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(&b_hi[ll][tj], &b_lo[ll][tj], x);
        else if (ti > l)
          daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
                &b_hi[ll][tj], &b_lo[ll][tj], x);
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti >= l++)
        daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
              &b_hi[ll][tj], &b_lo[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(&b_hi[ll][tj], &b_lo[ll][tj], x);
      else if (ti > l)
        daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
              &b_hi[ll][tj], &b_lo[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    dscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ int b_hi[kb][nb];
  __shared__ int b_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(A[0], b_hi[l], b_lo[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          daxpy(A[0], b_hi[ll], b_lo[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(b_hi[ll], b_lo[ll], x);
        else if (ti > l)
          daxpy(A[0], b_hi[ll], b_lo[ll], x);
        A += lda;
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti >= l++)
        daxpy(A[0], b_hi[ll], b_lo[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(b_hi[ll], b_lo[ll], x);
      else if (ti > l)
        daxpy(A[0], b_hi[ll], b_lo[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ int a_hi[mb][kb + 1];
  __shared__ int a_lo[mb][kb + 1];
  __shared__ int b_hi[kb][nb];
  __shared__ int b_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(A[i * lda]);
      a_lo[i + threadIdx.y][threadIdx.x] = __double2loint(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
                &b_hi[ll][tj], &b_lo[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(&b_hi[ll][tj], &b_lo[ll][tj], x);
        else if (ti < l)
          daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
                &b_hi[ll][tj], &b_lo[ll][tj], x);
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti <= l++)
        daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
              &b_hi[ll][tj], &b_lo[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(&b_hi[ll][tj], &b_lo[ll][tj], x);
      else if (ti < l)
        daxpy(__hiloint2double(a_hi[ti][ll], a_lo[ti][ll]),
              &b_hi[ll][tj], &b_lo[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for DGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(A[i * lda]);
      a_lo[i + threadIdx.y][threadIdx.x] = __double2loint(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(B[j * ldb]);
      b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      daxpy(__hiloint2double(a_hi[ti][l], a_lo[ti][l]),
            &b_hi[l][tj], &b_lo[l][tj], x);

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    daxpy(__hiloint2double(a_hi[ti][l], a_lo[ti][l]),
          &b_hi[l][tj], &b_lo[l][tj], x);

  if (m - bi - ti > 0)
    dscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] += B[0]; \
      daxpy(8 - (i) - 1, B[0], &a_hi[(i)][(i) + 1], &a_lo[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      daxpy(8 - (i), B[0], &a_hi[(i)][(i)], &a_lo[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      daxpy((i) - 1, B[0], a_hi[(i) - 1], a_lo[(i) - 1], x); \
      x[(i) - 1] += B[0]; \
    } \
    else \
      daxpy((i), B[0], a_hi[(i) - 1], a_lo[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_hi[kb][nb];
  __shared__ int a_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(A[j * lda]);
      a_lo[threadIdx.x][j + threadIdx.y] = __double2loint(A[j * lda]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a_hi[l], a_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(A[j * lda]);
      a_lo[threadIdx.x][j + threadIdx.y] = __double2loint(A[j * lda]);
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC( 0); B += ldb; INNER_RIGHT_LOOP_DEC( 1); B += ldb;
    INNER_RIGHT_LOOP_DEC( 2); B += ldb; INNER_RIGHT_LOOP_DEC( 3); B += ldb;
    INNER_RIGHT_LOOP_DEC( 4); B += ldb; INNER_RIGHT_LOOP_DEC( 5); B += ldb;
    INNER_RIGHT_LOOP_DEC( 6); B += ldb; INNER_RIGHT_LOOP_DEC( 7); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb;
  if (k > 1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb;
  if (k > 2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb;
  if (k > 3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb;
  if (k > 4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb;
  if (k > 5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb;
  if (k > 6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb;
  if (k > 7) { INNER_RIGHT_LOOP_DEC( 7); }}}}}}}}

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_hi[kb][nb];
  __shared__ int a_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(A[l * lda]);
      a_lo[l + threadIdx.y][threadIdx.x] = __double2loint(A[l * lda]);
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC( 1); B += ldb;
    INNER_RIGHT_LOOP_INC( 2); B += ldb;
    INNER_RIGHT_LOOP_INC( 3); B += ldb;
    INNER_RIGHT_LOOP_INC( 4); B += ldb;
    INNER_RIGHT_LOOP_INC( 5); B += ldb;
    INNER_RIGHT_LOOP_INC( 6); B += ldb;
    INNER_RIGHT_LOOP_INC( 7); B += ldb;
    INNER_RIGHT_LOOP_INC( 8); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb;
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb;
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb;
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb;
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb;
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb;
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb;
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); }}}}}}}}

  // Process non-diagonal blocks as for DGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(A[l * lda]);
      a_lo[l + threadIdx.y][threadIdx.x] = __double2loint(A[l * lda]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a_hi[l], a_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(B[0], a_hi[l], a_lo[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLN(int m, int n,
                        double alpha, const double * __restrict__ A, int lda,
                        const double * __restrict__ B, int ldb,
                        double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_hi[kb][nb];
  __shared__ int a_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(A[j * lda]);
      a_lo[threadIdx.x][j + threadIdx.y] = __double2loint(A[j * lda]);
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC( 1); B += ldb;
    INNER_RIGHT_LOOP_INC( 2); B += ldb;
    INNER_RIGHT_LOOP_INC( 3); B += ldb;
    INNER_RIGHT_LOOP_INC( 4); B += ldb;
    INNER_RIGHT_LOOP_INC( 5); B += ldb;
    INNER_RIGHT_LOOP_INC( 6); B += ldb;
    INNER_RIGHT_LOOP_INC( 7); B += ldb;
    INNER_RIGHT_LOOP_INC( 8); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb;
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb;
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb;
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb;
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb;
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb;
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb;
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); }}}}}}}}

  // Process non-diagonal blocks as for DGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(A[j * lda]);
      a_lo[threadIdx.x][j + threadIdx.y] = __double2loint(A[j * lda]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a_hi[l], a_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(B[0], a_hi[l], a_lo[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_hi[kb][nb];
  __shared__ int a_lo[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(A[l * lda]);
      a_lo[l + threadIdx.y][threadIdx.x] = __double2loint(A[l * lda]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a_hi[l], a_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(A[l * lda]);
      a_lo[l + threadIdx.y][threadIdx.x] = __double2loint(A[l * lda]);
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC( 0); B += ldb; INNER_RIGHT_LOOP_DEC( 1); B += ldb;
    INNER_RIGHT_LOOP_DEC( 2); B += ldb; INNER_RIGHT_LOOP_DEC( 3); B += ldb;
    INNER_RIGHT_LOOP_DEC( 4); B += ldb; INNER_RIGHT_LOOP_DEC( 5); B += ldb;
    INNER_RIGHT_LOOP_DEC( 6); B += ldb; INNER_RIGHT_LOOP_DEC( 7); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb;
  if (k > 1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb;
  if (k > 2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb;
  if (k > 3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb;
  if (k > 4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb;
  if (k > 5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb;
  if (k > 6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb;
  if (k > 7) { INNER_RIGHT_LOOP_DEC( 7); }}}}}}}}

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

#else

// y(1:8) += alpha * x(1:8)
__device__ void daxpy(double alpha, const double * x, double * y) {
  y[ 0] += alpha * x[ 0]; y[ 1] += alpha * x[ 1];
  y[ 2] += alpha * x[ 2]; y[ 3] += alpha * x[ 3];
  y[ 4] += alpha * x[ 4]; y[ 5] += alpha * x[ 5];
  y[ 6] += alpha * x[ 6]; y[ 7] += alpha * x[ 7];
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

// y(1:n) = alpha * x(1:n)
__device__ void dscal(int n, double alpha, const double * x, double * y, int incy) {
  if (n <= 0) return;
  y[0] = alpha * x[ 0]; if ( 1 >= n) return; y += incy;
  y[0] = alpha * x[ 1]; if ( 2 >= n) return; y += incy;
  y[0] = alpha * x[ 2]; if ( 3 >= n) return; y += incy;
  y[0] = alpha * x[ 3]; if ( 4 >= n) return; y += incy;
  y[0] = alpha * x[ 4]; if ( 5 >= n) return; y += incy;
  y[0] = alpha * x[ 5]; if ( 6 >= n) return; y += incy;
  y[0] = alpha * x[ 6]; if ( 7 >= n) return; y += incy;
  y[0] = alpha * x[ 7];
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ double b[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          daxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(b[ll], x);
        else if (ti < l)
          daxpy(A[0], b[ll], x);
        A += lda;
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti <= l++)
        daxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(b[ll], x);
      else if (ti < l)
        daxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for DGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(A[0], b[l], x);
      A += lda;
    }

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(A[0], b[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ double a[mb][kb + 1];
  __shared__ double b[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      daxpy(a[ti][l], &b[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          daxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(&b[ll][tj], x);
        else if (ti > l)
          daxpy(a[ti][ll], &b[ll][tj], x);
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti >= l++)
        daxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(&b[ll][tj], x);
      else if (ti > l)
        daxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    dscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ double b[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(A[0], b[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          daxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(b[ll], x);
        else if (ti > l)
          daxpy(A[0], b[ll], x);
        A += lda;
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti >= l++)
        daxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(b[ll], x);
      else if (ti > l)
        daxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ double a[mb][kb + 1];
  __shared__ double b[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          daxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          daxpy(&b[ll][tj], x);
        else if (ti < l)
          daxpy(a[ti][ll], &b[ll][tj], x);
        l++;
      }
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (diag == CBlasNonUnit) {
    for (int ll = 0; ll < k; ll++) {
      if (ti <= l++)
        daxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        daxpy(&b[ll][tj], x);
      else if (ti < l)
        daxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for SGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      daxpy(a[ti][l], &b[l][tj], x);

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    daxpy(a[ti][l], &b[l][tj], x);

  if (m - bi - ti > 0)
    dscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] += B[0]; \
      daxpy(8 - (i) - 1, B[0], &a[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      daxpy(8 - (i), B[0], &a[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      daxpy((i) - 1, B[0], a[(i) - 1], x); \
      x[(i) - 1] += B[0]; \
    } \
    else \
      daxpy((i), B[0], a[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ double a[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC( 0); B += ldb; INNER_RIGHT_LOOP_DEC( 1); B += ldb;
    INNER_RIGHT_LOOP_DEC( 2); B += ldb; INNER_RIGHT_LOOP_DEC( 3); B += ldb;
    INNER_RIGHT_LOOP_DEC( 4); B += ldb; INNER_RIGHT_LOOP_DEC( 5); B += ldb;
    INNER_RIGHT_LOOP_DEC( 6); B += ldb; INNER_RIGHT_LOOP_DEC( 7); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb;
  if (k > 1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb;
  if (k > 2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb;
  if (k > 3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb;
  if (k > 4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb;
  if (k > 5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb;
  if (k > 6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb;
  if (k > 7) { INNER_RIGHT_LOOP_DEC( 7); }}}}}}}}

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ double a[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC( 1); B += ldb;
    INNER_RIGHT_LOOP_INC( 2); B += ldb;
    INNER_RIGHT_LOOP_INC( 3); B += ldb;
    INNER_RIGHT_LOOP_INC( 4); B += ldb;
    INNER_RIGHT_LOOP_INC( 5); B += ldb;
    INNER_RIGHT_LOOP_INC( 6); B += ldb;
    INNER_RIGHT_LOOP_INC( 7); B += ldb;
    INNER_RIGHT_LOOP_INC( 8); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb;
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb;
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb;
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb;
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb;
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb;
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb;
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); }}}}}}}}

  // Process non-diagonal blocks as for DGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLN(int m, int n,
                        double alpha, const double * __restrict__ A, int lda,
                        const double * __restrict__ B, int ldb,
                        double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ double a[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC( 1); B += ldb;
    INNER_RIGHT_LOOP_INC( 2); B += ldb;
    INNER_RIGHT_LOOP_INC( 3); B += ldb;
    INNER_RIGHT_LOOP_INC( 4); B += ldb;
    INNER_RIGHT_LOOP_INC( 5); B += ldb;
    INNER_RIGHT_LOOP_INC( 6); B += ldb;
    INNER_RIGHT_LOOP_INC( 7); B += ldb;
    INNER_RIGHT_LOOP_INC( 8); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb;
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb;
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb;
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb;
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb;
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb;
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb;
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); }}}}}}}}

  // Process non-diagonal blocks as for DGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    daxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ double a[kb][nb];

  double x[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  // Process non-diagonal blocks as for DGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      daxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC( 0); B += ldb; INNER_RIGHT_LOOP_DEC( 1); B += ldb;
    INNER_RIGHT_LOOP_DEC( 2); B += ldb; INNER_RIGHT_LOOP_DEC( 3); B += ldb;
    INNER_RIGHT_LOOP_DEC( 4); B += ldb; INNER_RIGHT_LOOP_DEC( 5); B += ldb;
    INNER_RIGHT_LOOP_DEC( 6); B += ldb; INNER_RIGHT_LOOP_DEC( 7); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb;
  if (k > 1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb;
  if (k > 2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb;
  if (k > 3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb;
  if (k > 4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb;
  if (k > 5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb;
  if (k > 6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb;
  if (k > 7) { INNER_RIGHT_LOOP_DEC( 7); }}}}}}}}

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

#endif

template void dtrmmLUN<CBlasUnit,    64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLUN<CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLUT<CBlasUnit,    32, 16,  8,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLUT<CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLLN<CBlasUnit,    64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLLN<CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLLT<CBlasUnit,    32, 16,  8,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmLLT<CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);

template void dtrmmRUN<CBlasUnit,    64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRUN<CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRUT<CBlasUnit,    64,  8, 16,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRUT<CBlasNonUnit, 64,  8, 16,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRLN<CBlasUnit,    64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRLN<CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRLT<CBlasUnit,    64,  8, 16,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
template void dtrmmRLT<CBlasNonUnit, 64,  8, 16,  8,  8>(int, int, double, const double * __restrict__, int, const double * __restrict__, int, double * __restrict__, int);
