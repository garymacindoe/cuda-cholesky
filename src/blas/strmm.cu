#include "blas.h"

__device__ void saxpy(float alpha, const float * x, float * y) {
  y[ 0] += alpha * x[ 0]; y[ 1] += alpha * x[ 1]; y[ 2] += alpha * x[ 2]; y[ 3] += alpha * x[ 3];
  y[ 4] += alpha * x[ 4]; y[ 5] += alpha * x[ 5]; y[ 6] += alpha * x[ 6]; y[ 7] += alpha * x[ 7];
  y[ 8] += alpha * x[ 8]; y[ 9] += alpha * x[ 9]; y[10] += alpha * x[10]; y[11] += alpha * x[11];
  y[12] += alpha * x[12]; y[13] += alpha * x[13]; y[14] += alpha * x[14]; y[15] += alpha * x[15];
}

__device__ int min(int a, int b) { return (a < b) ? a : b; }

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmm2L(int m, int n,
                        float alpha, const float * __restrict__ A, int lda, const float * __restrict__ B, int ldb,
                        float * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (trans != CBlasNoTrans) {
    tj = 16 * (ti / mb);
    ti = ti % mb;
  }

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? bi * lda + bi + ti : bi + ti;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + bi + threadIdx.x : (bj + threadIdx.y) * ldb + threadIdx.x;
  }
  else {
    A += (uplo == CBlasUpper) ? (bi + threadIdx.y) * lda + threadIdx.x : (bi + threadIdx.y) * lda + bi + threadIdx.x;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + threadIdx.x : (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  }
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a[mb][kb + 1];
  __shared__ float b[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {
    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += bx) {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][l + threadIdx.x] = A[i * lda + l];
        }
        A += kb;
      }

#pragma unroll
      for (int j = 0; j < nb; j += by)
        b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti <= l++)
            saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            saxpy(1.0f, (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti < l)
            saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
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
          saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          saxpy(1.0f, (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti < l)
          saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for SGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? m - bi - mb : bi)
                                  : ((uplo == CBlasUpper) ? bi : m - bi - mb);
  while (k > 0) {
    if (trans != CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][l + threadIdx.x] = A[i * lda + l];
      }
      A += kb;
    }

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l++) {
        saxpy(A[0], b[l], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l++)
        saxpy(a[ti][l], &b[l][tj], x);
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      saxpy(A[0], b[l], x);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      saxpy(a[ti][l], &b[l][tj], x);
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {

    __syncthreads();

    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][threadIdx.x] = A[i * lda];
        A += kb;
      }

#pragma unroll
      for (int j = 0; j < nb; j += by)
        b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti >= l++)
            saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            saxpy(1.0f, (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti > l)
            saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
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
          saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          saxpy(1.0f, (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti > l)
          saxpy((trans == CBlasNoTrans) ? A[0] : a[ti][ll], (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }
  }

  n -= bj + tj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = alpha * x[ 0]; if ( 1 >= n) return; X += ldx;
  X[0] = alpha * x[ 1]; if ( 2 >= n) return; X += ldx;
  X[0] = alpha * x[ 2]; if ( 3 >= n) return; X += ldx;
  X[0] = alpha * x[ 3]; if ( 4 >= n) return; X += ldx;
  X[0] = alpha * x[ 4]; if ( 5 >= n) return; X += ldx;
  X[0] = alpha * x[ 5]; if ( 6 >= n) return; X += ldx;
  X[0] = alpha * x[ 6]; if ( 7 >= n) return; X += ldx;
  X[0] = alpha * x[ 7]; if ( 8 >= n) return; X += ldx;
  X[0] = alpha * x[ 8]; if ( 9 >= n) return; X += ldx;
  X[0] = alpha * x[ 9]; if (10 >= n) return; X += ldx;
  X[0] = alpha * x[10]; if (11 >= n) return; X += ldx;
  X[0] = alpha * x[11]; if (12 >= n) return; X += ldx;
  X[0] = alpha * x[12]; if (13 >= n) return; X += ldx;
  X[0] = alpha * x[13]; if (14 >= n) return; X += ldx;
  X[0] = alpha * x[14]; if (15 >= n) return; X += ldx;
  X[0] = alpha * x[15];
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strmm2R(int m, int n,
                        float alpha, const float * __restrict__ A, int lda, const float * __restrict__ B, int ldb,
                        float * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + threadIdx.x : (bj + threadIdx.y) * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bi + ti : bj * ldb + bi + ti;
  }
  else {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + bj + threadIdx.x : threadIdx.y * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bj * ldb + bi + ti : bi + ti;
  }
  X += bj * ldx + bi + ti;

  __shared__ float a[kb][nb];

  float x[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {
    int k = min(n - bj, nb);
    int l = 0;
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          a[threadIdx.x][j + threadIdx.y] = A[j * lda];
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] = A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti <= l++)
            saxpy(B[0], a[ll], x);
          B += ldb;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            saxpy(1.0f, a[ll], x);
          else if (ti < l)
            saxpy(B[0], a[ll], x);
          B += ldb;
          l++;
        }
      }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    if (diag == CBlasNonUnit) {
      for (int ll = 0; ll < k; ll++) {
        if (ti <= l++)
          saxpy(B[0], a[ll], x);
        B += ldb;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          saxpy(1.0f, a[ll], x);
        else if (ti < l)
          saxpy(B[0], a[ll], x);
        B += ldb;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for SGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? n - bj - nb : bj)
                                  : ((uplo == CBlasUpper) ? bj : n - bj - nb);
  while (k > 0) {
    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by)
        a[threadIdx.x][j + threadIdx.y] = A[j * lda];
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by)
        a[l + threadIdx.y][threadIdx.x] = A[l * lda];
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      saxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += (trans == CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    saxpy(B[0], a[l], x);
    B += ldb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {

    __syncthreads();

    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          a[threadIdx.x][j + threadIdx.y] = A[j * lda];
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] = A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti >= l++)
            saxpy(B[0], a[ll], x);
          B += ldb;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            saxpy(1.0f, a[ll], x);
          else if (ti > l)
            saxpy(B[0], a[ll], x);
          B += ldb;
          l++;
        }
      }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    if (diag == CBlasNonUnit) {
      for (int ll = 0; ll < k; ll++) {
        if (ti >= l++)
          saxpy(B[0], a[ll], x);
        B += ldb;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          saxpy(1.0f, a[ll], x);
        else if (ti > l)
          saxpy(B[0], a[ll], x);
        B += ldb;
        l++;
      }
    }
  }

  n -= bj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = alpha * x[ 0]; if ( 1 >= n) return; X += ldx;
  X[0] = alpha * x[ 1]; if ( 2 >= n) return; X += ldx;
  X[0] = alpha * x[ 2]; if ( 3 >= n) return; X += ldx;
  X[0] = alpha * x[ 3]; if ( 4 >= n) return; X += ldx;
  X[0] = alpha * x[ 4]; if ( 5 >= n) return; X += ldx;
  X[0] = alpha * x[ 5]; if ( 6 >= n) return; X += ldx;
  X[0] = alpha * x[ 6]; if ( 7 >= n) return; X += ldx;
  X[0] = alpha * x[ 7]; if ( 8 >= n) return; X += ldx;
  X[0] = alpha * x[ 8]; if ( 9 >= n) return; X += ldx;
  X[0] = alpha * x[ 9]; if (10 >= n) return; X += ldx;
  X[0] = alpha * x[10]; if (11 >= n) return; X += ldx;
  X[0] = alpha * x[11]; if (12 >= n) return; X += ldx;
  X[0] = alpha * x[12]; if (13 >= n) return; X += ldx;
  X[0] = alpha * x[13]; if (14 >= n) return; X += ldx;
  X[0] = alpha * x[14]; if (15 >= n) return; X += ldx;
  X[0] = alpha * x[15];
}

template void strmm2L<CBlasUpper, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasUpper, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasUpper, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasUpper, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasLower, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasLower, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasLower, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2L<CBlasLower, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasUpper, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasUpper, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasUpper, CBlasTrans,   CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasUpper, CBlasTrans,   CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasLower, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasLower, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasLower, CBlasTrans,   CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
template void strmm2R<CBlasLower, CBlasTrans,   CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, const float * __restrict__, int, float * __restrict__, int);
