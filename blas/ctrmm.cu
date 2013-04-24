#include "blas.h"
#include "caxpy.cu"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:n) = alpha * x(1:n)
__device__ void cscal(int n, cuComplex alpha, const cuComplex * __restrict__ x,
                      cuComplex * __restrict__ y, int incy) {
  if (n <= 0) return;
  y[0] = cuCmulf(alpha, x[0]); if (1 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[1]); if (2 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[2]); if (3 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[3]); if (4 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[4]); if (5 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[5]); if (6 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[6]); if (7 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[7]);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLUN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ float b_real[kb][nb];
  __shared__ float b_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          caxpy(A[0], b_real[ll], b_imag[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(b_real[ll], b_imag[ll], x);
        else if (ti < l)
          caxpy(A[0], b_real[ll], b_imag[ll], x);
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
        caxpy(A[0], b_real[ll], b_imag[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(b_real[ll], b_imag[ll], x);
      else if (ti < l)
        caxpy(A[0], b_real[ll], b_imag[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for CGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(A[0], b_real[l], b_imag[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(A[0], b_real[l], b_imag[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLUT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a_real[mb][kb + 1];
  __shared__ float a_imag[mb][kb + 1];
  __shared__ float b_real[kb][nb];
  __shared__ float b_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
      a_imag[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[i * lda])
                                             :  cuCimagf(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]),
            &b_real[l][tj], &b_imag[l][tj], x);

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
      a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
      a_imag[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[i * lda])
                                             :  cuCimagf(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                &b_real[ll][tj], &b_imag[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(&b_real[ll][tj], &b_imag[ll][tj], x);
        else if (ti > l)
          caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                &b_real[ll][tj], &b_imag[ll][tj], x);
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
        caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
              &b_real[ll][tj], &b_imag[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(&b_real[ll][tj], &b_imag[ll][tj], x);
      else if (ti > l)
        caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
              &b_real[ll][tj], &b_imag[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    cscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLLN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ float b_real[kb][nb];
  __shared__ float b_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(A[0], b_real[l], b_imag[l], x);
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
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti >= l++)
          caxpy(A[0], b_real[ll], b_imag[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(b_real[ll], b_imag[ll], x);
        else if (ti > l)
          caxpy(A[0], b_real[ll], b_imag[ll], x);
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
        caxpy(A[0], b_real[ll], b_imag[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(b_real[ll], b_imag[ll], x);
      else if (ti > l)
        caxpy(A[0], b_real[ll], b_imag[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLLT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a_real[mb][kb + 1];
  __shared__ float a_imag[mb][kb + 1];
  __shared__ float b_real[kb][nb];
  __shared__ float b_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
      a_imag[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[i * lda])
                                             :  cuCimagf(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (diag == CBlasNonUnit) {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti <= l++)
          caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                &b_real[ll][tj], &b_imag[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(&b_real[ll][tj], &b_imag[ll][tj], x);
        else if (ti < l)
          caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                &b_real[ll][tj], &b_imag[ll][tj], x);
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
        caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
              &b_real[ll][tj], &b_imag[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(&b_real[ll][tj], &b_imag[ll][tj], x);
      else if (ti < l)
        caxpy(make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
              &b_real[ll][tj], &b_imag[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for CGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
      a_imag[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[i * lda])
                                             :  cuCimagf(A[i * lda]);
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]),
            &b_real[l][tj], &b_imag[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]),
          &b_real[l][tj], &b_imag[l][tj], x);

  if (m - bi - ti > 0)
    cscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] = cuCaddf(x[(i)], B[0]); \
      caxpy(8 - (i) - 1, B[0], &a_real[(i)][(i) + 1], &a_imag[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      caxpy(8 - (i), B[0], &a_real[(i)][(i)], &a_imag[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      caxpy((i) - 1, B[0], a_real[(i) - 1], a_imag[(i) - 1], x); \
      x[(i) - 1] = cuCaddf(x[(i) - 1], B[0]); \
    } \
    else \
      caxpy((i), B[0], a_real[(i) - 1], a_imag[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRUN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a_real[kb][nb];
  __shared__ float a_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real[threadIdx.x][j + threadIdx.y] = cuCrealf(A[j * lda]);
      a_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(A[j * lda]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a_real[l], a_imag[l], x);
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
      a_real[threadIdx.x][j + threadIdx.y] = cuCrealf(A[j * lda]);
      a_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(A[j * lda]);
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;
    INNER_RIGHT_LOOP_DEC(4); B += ldb; INNER_RIGHT_LOOP_DEC(5); B += ldb;
    INNER_RIGHT_LOOP_DEC(6); B += ldb; INNER_RIGHT_LOOP_DEC(7); B += ldb;

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
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRUT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a_real[kb][nb];
  __shared__ float a_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real[l + threadIdx.y][threadIdx.x] = cuCrealf(A[l * lda]);
      a_imag[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[l * lda])
                                             :  cuCimagf(A[l * lda]);
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

  // Process non-diagonal blocks as for CGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real[l + threadIdx.y][threadIdx.x] = cuCrealf(A[l * lda]);
      a_imag[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[l * lda])
                                             :  cuCimagf(A[l * lda]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a_real[l], a_imag[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a_real[l], a_imag[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRLN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a_real[kb][nb];
  __shared__ float a_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real[threadIdx.x][j + threadIdx.y] = cuCrealf(A[j * lda]);
      a_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(A[j * lda]);
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

  // Process non-diagonal blocks as for CGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real[threadIdx.x][j + threadIdx.y] = cuCrealf(A[j * lda]);
      a_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(A[j * lda]);
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a_real[l], a_imag[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a_real[l], a_imag[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRLT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a_real[kb][nb];
  __shared__ float a_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real[l + threadIdx.y][threadIdx.x] = cuCrealf(A[l * lda]);
      a_imag[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[l * lda])
                                             :  cuCimagf(A[l * lda]);
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a_real[l], a_imag[l], x);
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
      a_real[l + threadIdx.y][threadIdx.x] = cuCrealf(A[l * lda]);
      a_imag[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                             ? -cuCimagf(A[l * lda])
                                             :  cuCimagf(A[l * lda]);
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
    cscal(n - bj, alpha, x, X, ldx);
}

#else

// y(1:n) = alpha * x(1:n)
__device__ void cscal(int n, cuComplex alpha, const cuComplex * x, cuComplex * y, int incy) {
  if (n <= 0) return;
  y[0] = cuCmulf(alpha, x[0]); if (1 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[1]); if (2 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[2]); if (3 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[3]); if (4 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[4]); if (5 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[5]); if (6 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[6]); if (7 >= n) return; y += incy;
  y[0] = cuCmulf(alpha, x[7]);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLUN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex b[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

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
          caxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(b[ll], x);
        else if (ti < l)
          caxpy(A[0], b[ll], x);
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
        caxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(b[ll], x);
      else if (ti < l)
        caxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for CGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(A[0], b[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(A[0], b[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLUT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ cuComplex a[mb][kb + 1];
  __shared__ cuComplex b[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[i * lda]) : A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      caxpy(a[ti][l], &b[l][tj], x);

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
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[i * lda]) : A[i * lda];
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
          caxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(&b[ll][tj], x);
        else if (ti > l)
          caxpy(a[ti][ll], &b[ll][tj], x);
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
        caxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(&b[ll][tj], x);
      else if (ti > l)
        caxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    cscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLLN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex b[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(A[0], b[l], x);
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
          caxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(b[ll], x);
        else if (ti > l)
          caxpy(A[0], b[ll], x);
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
        caxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(b[ll], x);
      else if (ti > l)
        caxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmLLT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 8 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ cuComplex a[mb][kb + 1];
  __shared__ cuComplex b[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[i * lda]) : A[i * lda];
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
          caxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          caxpy(&b[ll][tj], x);
        else if (ti < l)
          caxpy(a[ti][ll], &b[ll][tj], x);
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
        caxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        caxpy(&b[ll][tj], x);
      else if (ti < l)
        caxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for CGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[i * lda]) : A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      caxpy(a[ti][l], &b[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    caxpy(a[ti][l], &b[l][tj], x);

  if (m - bi - ti > 0)
    cscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] = cuCaddf(x[(i)], B[0]); \
      caxpy(8 - (i) - 1, B[0], &a[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      caxpy(8 - (i), B[0], &a[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      caxpy((i) - 1, B[0], a[(i) - 1], x); \
      x[(i) - 1] = cuCaddf(x[(i) - 1], B[0]); \
    } \
    else \
      caxpy((i), B[0], a[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRUN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex a[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a[l], x);
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

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;
    INNER_RIGHT_LOOP_DEC(4); B += ldb; INNER_RIGHT_LOOP_DEC(5); B += ldb;
    INNER_RIGHT_LOOP_DEC(6); B += ldb; INNER_RIGHT_LOOP_DEC(7); B += ldb;

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
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRUT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex a[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[l * lda]) : A[l * lda];

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

  // Process non-diagonal blocks as for CGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[l * lda]) : A[l * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRLN(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex a[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

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

  // Process non-diagonal blocks as for CGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    cscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmmRLT(const cuComplex * __restrict__ A,
                         const cuComplex * __restrict__ B, cuComplex * __restrict__ X,
                         cuComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuComplex a[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // Process non-diagonal blocks as for CGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[l * lda]) : A[l * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a[l], x);
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
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConjf(A[l * lda]) : A[l * lda];

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
    cscal(n - bj, alpha, x, X, ldx);
}

#endif

template __global__ void ctrmmLUN<CBlasUnit,    64,  8, 16, 16,  4>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLUN<CBlasNonUnit, 64,  8, 16, 16,  4>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLUT<CBlasTrans,     CBlasUnit,    32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLUT<CBlasTrans,     CBlasNonUnit, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLUT<CBlasConjTrans, CBlasUnit,    32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLUT<CBlasConjTrans, CBlasNonUnit, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLN<CBlasUnit,    64,  8, 16, 16,  4>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLN<CBlasNonUnit, 64,  8, 16, 16,  4>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLT<CBlasTrans,     CBlasUnit,    32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLT<CBlasTrans,     CBlasNonUnit, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLT<CBlasConjTrans, CBlasUnit,    32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmLLT<CBlasConjTrans, CBlasNonUnit, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);

template __global__ void ctrmmRUN<CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRUN<CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRUT<CBlasTrans,     CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRUT<CBlasTrans,     CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRUT<CBlasConjTrans, CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRUT<CBlasConjTrans, CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLN<CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLN<CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLT<CBlasTrans,     CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLT<CBlasTrans,     CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLT<CBlasConjTrans, CBlasUnit,    64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
template __global__ void ctrmmRLT<CBlasConjTrans, CBlasNonUnit, 64,  8,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, int, int, int, int, int);
