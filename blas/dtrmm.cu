#include "blas.h"
#include "daxpy.cu"

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:n) = alpha * x(1:n)
__device__ void dscal(int n, double alpha, const double * __restrict__ x,
                      double * __restrict__ y, int incy) {
  if (n <= 0) return;
  y[0] = alpha * x[0]; if (1 >= n) return; y += incy;
  y[0] = alpha * x[1]; if (2 >= n) return; y += incy;
  y[0] = alpha * x[2]; if (3 >= n) return; y += incy;
  y[0] = alpha * x[3]; if (4 >= n) return; y += incy;
  y[0] = alpha * x[4]; if (5 >= n) return; y += incy;
  y[0] = alpha * x[5]; if (6 >= n) return; y += incy;
  y[0] = alpha * x[6]; if (7 >= n) return; y += incy;
  y[0] = alpha * x[7];
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__device__ void dtrmmLUN(int m, int n,
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

    __syncthreads();

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
__device__ void dtrmmLUT(int m, int n,
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
__device__ void dtrmmLLN(int m, int n,
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
__device__ void dtrmmLLT(int m, int n,
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

    __syncthreads();

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
__device__ void dtrmmRUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

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

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_DEC(3); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_DEC(4); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_DEC(5); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_DEC(6); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_DEC(7);

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__device__ void dtrmmRUT(int m, int n,
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

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;
    INNER_RIGHT_LOOP_INC(5); B += ldb;
    INNER_RIGHT_LOOP_INC(6); B += ldb;
    INNER_RIGHT_LOOP_INC(7); B += ldb;
    INNER_RIGHT_LOOP_INC(8); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_INC(4); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_INC(5); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_INC(6); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_INC(7); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_INC(8);

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
__device__ void dtrmmRLN(int m, int n,
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

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;
    INNER_RIGHT_LOOP_INC(5); B += ldb;
    INNER_RIGHT_LOOP_INC(6); B += ldb;
    INNER_RIGHT_LOOP_INC(7); B += ldb;
    INNER_RIGHT_LOOP_INC(8); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_INC(4); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_INC(5); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_INC(6); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_INC(7); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_INC(8);

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
__device__ void dtrmmRLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

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

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;
    INNER_RIGHT_LOOP_DEC(4); B += ldb; INNER_RIGHT_LOOP_DEC(5); B += ldb;
    INNER_RIGHT_LOOP_DEC(6); B += ldb; INNER_RIGHT_LOOP_DEC(7); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_DEC(3); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_DEC(4); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_DEC(5); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_DEC(6); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_DEC(7);

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

#else

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
__device__ void dtrmmLUN(int m, int n,
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

    __syncthreads();

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
__device__ void dtrmmLUT(int m, int n,
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
__device__ void dtrmmLLN(int m, int n,
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
__device__ void dtrmmLLT(int m, int n,
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

  // Process any non-diagonal blocks as for DGEMM
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

    __syncthreads();

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
__device__ void dtrmmRUN(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

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

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_DEC(3); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_DEC(4); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_DEC(5); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_DEC(6); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_DEC(7);

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__device__ void dtrmmRUT(int m, int n,
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

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;
    INNER_RIGHT_LOOP_INC(5); B += ldb;
    INNER_RIGHT_LOOP_INC(6); B += ldb;
    INNER_RIGHT_LOOP_INC(7); B += ldb;
    INNER_RIGHT_LOOP_INC(8); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_INC(4); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_INC(5); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_INC(6); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_INC(7); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_INC(8);

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
__device__ void dtrmmRLN(int m, int n,
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

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;
    INNER_RIGHT_LOOP_INC(5); B += ldb;
    INNER_RIGHT_LOOP_INC(6); B += ldb;
    INNER_RIGHT_LOOP_INC(7); B += ldb;
    INNER_RIGHT_LOOP_INC(8); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_INC(4); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_INC(5); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_INC(6); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_INC(7); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_INC(8);

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
__device__ void dtrmmRLT(int m, int n,
                         double alpha, const double * __restrict__ A, int lda,
                         const double * __restrict__ B, int ldb,
                         double * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

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

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;
    INNER_RIGHT_LOOP_DEC(4); B += ldb; INNER_RIGHT_LOOP_DEC(5); B += ldb;
    INNER_RIGHT_LOOP_DEC(6); B += ldb; INNER_RIGHT_LOOP_DEC(7); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3) { INNER_RIGHT_LOOP_DEC(3); B += ldb; }
  if (k > 4) { INNER_RIGHT_LOOP_DEC(4); B += ldb; }
  if (k > 5) { INNER_RIGHT_LOOP_DEC(5); B += ldb; }
  if (k > 6) { INNER_RIGHT_LOOP_DEC(6); B += ldb; }
  if (k > 7)   INNER_RIGHT_LOOP_DEC(7);

  if (m - bi - ti > 0)
    dscal(n - bj, alpha, x, X, ldx);
}

#endif

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUN(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmLUN<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLUT(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmLUT<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLN(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmLLN<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmLLT(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmLLT<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUN(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmRUN<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRUT(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmRUT<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLN(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmRLN<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void dtrmmRLT(const double * __restrict__ A,
                         const double * __restrict__ B, double * __restrict__ X,
                         double alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {
  dtrmmRLT<diag, mb, nb, kb, bx, by>(m, n, alpha, A, lda, B, ldb, X, ldx);
}

template __global__ void dtrmmLUN<CBlasUnit,    64,  8, 16, 16,  4>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLUN<CBlasNonUnit, 64,  8, 16, 16,  4>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLUT<CBlasUnit,    32, 16,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLUT<CBlasNonUnit, 32, 16,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLLN<CBlasUnit,    64,  8, 16, 16,  4>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLLN<CBlasNonUnit, 64,  8, 16, 16,  4>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLLT<CBlasUnit,    32, 16,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmLLT<CBlasNonUnit, 32, 16,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);

template __global__ void dtrmmRUN<CBlasUnit,    64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRUN<CBlasNonUnit, 64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRUT<CBlasUnit,    64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRUT<CBlasNonUnit, 64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRLN<CBlasUnit,    64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRLN<CBlasNonUnit, 64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRLT<CBlasUnit,    64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
template __global__ void dtrmmRLT<CBlasNonUnit, 64,  8,  8,  8,  8>(const double * __restrict__, const double * __restrict__, double * __restrict__, double, int, int, int, int, int);
