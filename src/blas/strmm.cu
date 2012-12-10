#include "blas.h"

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

// y(1:n) = alpha * x(1:n)
__device__ void sscal(int n, float alpha, const float * __restrict__ x, float * __restrict__ y, int incy) {
  if (n <= 0) return;
  y[0] = alpha * x[ 0]; if ( 1 >= n) return; y += incy;
  y[0] = alpha * x[ 1]; if ( 2 >= n) return; y += incy;
  y[0] = alpha * x[ 2]; if ( 3 >= n) return; y += incy;
  y[0] = alpha * x[ 3]; if ( 4 >= n) return; y += incy;
  y[0] = alpha * x[ 4]; if ( 5 >= n) return; y += incy;
  y[0] = alpha * x[ 5]; if ( 6 >= n) return; y += incy;
  y[0] = alpha * x[ 6]; if ( 7 >= n) return; y += incy;
  y[0] = alpha * x[ 7]; if ( 8 >= n) return; y += incy;
  y[0] = alpha * x[ 8]; if ( 9 >= n) return; y += incy;
  y[0] = alpha * x[ 9]; if (10 >= n) return; y += incy;
  y[0] = alpha * x[10]; if (11 >= n) return; y += incy;
  y[0] = alpha * x[11]; if (12 >= n) return; y += incy;
  y[0] = alpha * x[12]; if (13 >= n) return; y += incy;
  y[0] = alpha * x[13]; if (14 >= n) return; y += incy;
  y[0] = alpha * x[14]; if (15 >= n) return; y += incy;
  y[0] = alpha * x[15];
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmLUN(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ float b[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

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
          saxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          saxpy(b[ll], x);
        else if (ti < l)
          saxpy(A[0], b[ll], x);
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
        saxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        saxpy(b[ll], x);
      else if (ti < l)
        saxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for SGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(A[0], b[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    saxpy(A[0], b[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmLUT(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  const int tj = 16 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a[mb][kb + 1];
  __shared__ float b[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // Process non-diagonal blocks as for SGEMM
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
      saxpy(a[ti][l], &b[l][tj], x);

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
          saxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          saxpy(&b[ll][tj], x);
        else if (ti > l)
          saxpy(a[ti][ll], &b[ll][tj], x);
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
        saxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        saxpy(&b[ll][tj], x);
      else if (ti > l)
        saxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    sscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmLLN(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ float b[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // Process non-diagonal blocks as for SGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(A[0], b[l], x);
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
          saxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          saxpy(b[ll], x);
        else if (ti > l)
          saxpy(A[0], b[ll], x);
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
        saxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        saxpy(b[ll], x);
      else if (ti > l)
        saxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmLLT(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  const int tj = 16 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a[mb][kb + 1];
  __shared__ float b[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

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
          saxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          saxpy(&b[ll][tj], x);
        else if (ti < l)
          saxpy(a[ti][ll], &b[ll][tj], x);
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
        saxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        saxpy(&b[ll][tj], x);
      else if (ti < l)
        saxpy(a[ti][ll], &b[ll][tj], x);
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
      saxpy(a[ti][l], &b[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    saxpy(a[ti][l], &b[l][tj], x);

  if (m - bi - ti > 0)
    sscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] += B[0]; \
      saxpy(16 - (i) - 1, B[0], &a[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      saxpy(16 - (i), B[0], &a[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      saxpy((i) - 1, B[0], a[(i) - 1], x); \
      x[(i) - 1] += B[0]; \
    } \
    else \
      saxpy((i), B[0], a[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmRUN(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // Process non-diagonal blocks as for SGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(B[0], a[l], x);
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
    INNER_RIGHT_LOOP_DEC( 8); B += ldb; INNER_RIGHT_LOOP_DEC( 9); B += ldb;
    INNER_RIGHT_LOOP_DEC(10); B += ldb; INNER_RIGHT_LOOP_DEC(11); B += ldb;
    INNER_RIGHT_LOOP_DEC(12); B += ldb; INNER_RIGHT_LOOP_DEC(13); B += ldb;
    INNER_RIGHT_LOOP_DEC(14); B += ldb; INNER_RIGHT_LOOP_DEC(15); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb; }
  if (k >  1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb; }
  if (k >  2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb; }
  if (k >  3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb; }
  if (k >  4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb; }
  if (k >  5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb; }
  if (k >  6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb; }
  if (k >  7) { INNER_RIGHT_LOOP_DEC( 7); B += ldb; }
  if (k >  8) { INNER_RIGHT_LOOP_DEC( 8); B += ldb; }
  if (k >  9) { INNER_RIGHT_LOOP_DEC( 9); B += ldb; }
  if (k > 10) { INNER_RIGHT_LOOP_DEC(10); B += ldb; }
  if (k > 11) { INNER_RIGHT_LOOP_DEC(11); B += ldb; }
  if (k > 12) { INNER_RIGHT_LOOP_DEC(12); B += ldb; }
  if (k > 13) { INNER_RIGHT_LOOP_DEC(13); B += ldb; }
  if (k > 14) { INNER_RIGHT_LOOP_DEC(14); B += ldb; }
  if (k > 15)   INNER_RIGHT_LOOP_DEC(15);

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmRUT(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

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
    INNER_RIGHT_LOOP_INC( 9); B += ldb;
    INNER_RIGHT_LOOP_INC(10); B += ldb;
    INNER_RIGHT_LOOP_INC(11); B += ldb;
    INNER_RIGHT_LOOP_INC(12); B += ldb;
    INNER_RIGHT_LOOP_INC(13); B += ldb;
    INNER_RIGHT_LOOP_INC(14); B += ldb;
    INNER_RIGHT_LOOP_INC(15); B += ldb;
    INNER_RIGHT_LOOP_INC(16); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb; }
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb; }
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb; }
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb; }
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb; }
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb; }
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb; }
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); B += ldb; }
  if (k >  8) { INNER_RIGHT_LOOP_INC( 9); B += ldb; }
  if (k >  9) { INNER_RIGHT_LOOP_INC(10); B += ldb; }
  if (k > 10) { INNER_RIGHT_LOOP_INC(11); B += ldb; }
  if (k > 11) { INNER_RIGHT_LOOP_INC(12); B += ldb; }
  if (k > 12) { INNER_RIGHT_LOOP_INC(13); B += ldb; }
  if (k > 13) { INNER_RIGHT_LOOP_INC(14); B += ldb; }
  if (k > 14) { INNER_RIGHT_LOOP_INC(15); B += ldb; }
  if (k > 15) { INNER_RIGHT_LOOP_INC(16); B += ldb; }

  // Process non-diagonal blocks as for SGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    saxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmRLN(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

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
    INNER_RIGHT_LOOP_INC( 9); B += ldb;
    INNER_RIGHT_LOOP_INC(10); B += ldb;
    INNER_RIGHT_LOOP_INC(11); B += ldb;
    INNER_RIGHT_LOOP_INC(12); B += ldb;
    INNER_RIGHT_LOOP_INC(13); B += ldb;
    INNER_RIGHT_LOOP_INC(14); B += ldb;
    INNER_RIGHT_LOOP_INC(15); B += ldb;
    INNER_RIGHT_LOOP_INC(16); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_INC( 1); B += ldb; }
  if (k >  1) { INNER_RIGHT_LOOP_INC( 2); B += ldb; }
  if (k >  2) { INNER_RIGHT_LOOP_INC( 3); B += ldb; }
  if (k >  3) { INNER_RIGHT_LOOP_INC( 4); B += ldb; }
  if (k >  4) { INNER_RIGHT_LOOP_INC( 5); B += ldb; }
  if (k >  5) { INNER_RIGHT_LOOP_INC( 6); B += ldb; }
  if (k >  6) { INNER_RIGHT_LOOP_INC( 7); B += ldb; }
  if (k >  7) { INNER_RIGHT_LOOP_INC( 8); B += ldb; }
  if (k >  8) { INNER_RIGHT_LOOP_INC( 9); B += ldb; }
  if (k >  9) { INNER_RIGHT_LOOP_INC(10); B += ldb; }
  if (k > 10) { INNER_RIGHT_LOOP_INC(11); B += ldb; }
  if (k > 11) { INNER_RIGHT_LOOP_INC(12); B += ldb; }
  if (k > 12) { INNER_RIGHT_LOOP_INC(13); B += ldb; }
  if (k > 13) { INNER_RIGHT_LOOP_INC(14); B += ldb; }
  if (k > 14) { INNER_RIGHT_LOOP_INC(15); B += ldb; }
  if (k > 15) { INNER_RIGHT_LOOP_INC(16); B += ldb; }

  // Process non-diagonal blocks as for SGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    saxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmmRLT(const float * __restrict__ A,
                         const float * __restrict__ B, float * __restrict__ X,
                         float alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ float a[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // Process non-diagonal blocks as for SGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = A[l * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(B[0], a[l], x);
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
    INNER_RIGHT_LOOP_DEC( 8); B += ldb; INNER_RIGHT_LOOP_DEC( 9); B += ldb;
    INNER_RIGHT_LOOP_DEC(10); B += ldb; INNER_RIGHT_LOOP_DEC(11); B += ldb;
    INNER_RIGHT_LOOP_DEC(12); B += ldb; INNER_RIGHT_LOOP_DEC(13); B += ldb;
    INNER_RIGHT_LOOP_DEC(14); B += ldb; INNER_RIGHT_LOOP_DEC(15); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k >  0) { INNER_RIGHT_LOOP_DEC( 0); B += ldb; }
  if (k >  1) { INNER_RIGHT_LOOP_DEC( 1); B += ldb; }
  if (k >  2) { INNER_RIGHT_LOOP_DEC( 2); B += ldb; }
  if (k >  3) { INNER_RIGHT_LOOP_DEC( 3); B += ldb; }
  if (k >  4) { INNER_RIGHT_LOOP_DEC( 4); B += ldb; }
  if (k >  5) { INNER_RIGHT_LOOP_DEC( 5); B += ldb; }
  if (k >  6) { INNER_RIGHT_LOOP_DEC( 6); B += ldb; }
  if (k >  7) { INNER_RIGHT_LOOP_DEC( 7); B += ldb; }
  if (k >  8) { INNER_RIGHT_LOOP_DEC( 8); B += ldb; }
  if (k >  9) { INNER_RIGHT_LOOP_DEC( 9); B += ldb; }
  if (k > 10) { INNER_RIGHT_LOOP_DEC(10); B += ldb; }
  if (k > 11) { INNER_RIGHT_LOOP_DEC(11); B += ldb; }
  if (k > 12) { INNER_RIGHT_LOOP_DEC(12); B += ldb; }
  if (k > 13) { INNER_RIGHT_LOOP_DEC(13); B += ldb; }
  if (k > 14) { INNER_RIGHT_LOOP_DEC(14); B += ldb; }
  if (k > 15)   INNER_RIGHT_LOOP_DEC(15);

  if (m - bi - ti > 0)
    sscal(n - bj, alpha, x, X, ldx);
}

template void strmmLUN<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLUN<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLUT<CBlasUnit,    32, 32,  8,  8,  8>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLUT<CBlasNonUnit, 32, 32,  8,  8,  8>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLLN<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLLN<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLLT<CBlasUnit,    32, 32,  8,  8,  8>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmLLT<CBlasNonUnit, 32, 32,  8,  8,  8>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);

template void strmmRUN<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRUN<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRUT<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRUT<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRLN<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRLN<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRLT<CBlasUnit,    64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
template void strmmRLT<CBlasNonUnit, 64, 16, 16, 16,  4>(const float * __restrict__, const float * __restrict__, float * __restrict__, float, int, int, int, int, int);
