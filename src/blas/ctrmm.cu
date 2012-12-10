#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICTS__)

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

// y(1:n) += alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const float * x_real, const float * x_imag, cuComplex * y) {
  y[0] = cuCfmaf(alpha, make_cuComplex(x_real[0], x_imag[0]), y[0]); if (1 >= n) return;
  y[1] = cuCfmaf(alpha, make_cuComplex(x_real[1], x_imag[1]), y[1]); if (2 >= n) return;
  y[2] = cuCfmaf(alpha, make_cuComplex(x_real[2], x_imag[2]), y[2]); if (3 >= n) return;
  y[3] = cuCfmaf(alpha, make_cuComplex(x_real[3], x_imag[3]), y[3]); if (4 >= n) return;
  y[4] = cuCfmaf(alpha, make_cuComplex(x_real[4], x_imag[4]), y[4]); if (5 >= n) return;
  y[5] = cuCfmaf(alpha, make_cuComplex(x_real[5], x_imag[5]), y[5]); if (6 >= n) return;
  y[6] = cuCfmaf(alpha, make_cuComplex(x_real[6], x_imag[6]), y[6]); if (7 >= n) return;
  y[7] = cuCfmaf(alpha, make_cuComplex(x_real[7], x_imag[7]), y[7]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmm2L(int m, int n,
                        cuComplex alpha, const cuComplex * __restrict__ A, int lda, const cuComplex * __restrict__ B, int ldb,
                        cuComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (trans != CBlasNoTrans) {
    tj = 8 * (ti / mb);
    ti = ti % mb;
  }

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? bi * lda + bi + ti : bi + ti;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + bi + threadIdx.x
                              : (bj + threadIdx.y) * ldb + threadIdx.x;
  }
  else {
    A += (uplo == CBlasUpper) ? (bi + threadIdx.y) * lda + threadIdx.x
                              : (bi + threadIdx.y) * lda + bi + threadIdx.x;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + threadIdx.x
                              : (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  }
  X += (bj + tj) * ldx + bi + ti;

  __shared__ float a_real[mb][kb + 1];
  __shared__ float a_imag[mb][kb + 1];
  __shared__ float b_real[kb][nb];
  __shared__ float b_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {
    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
        if (trans == CBlasConjTrans) {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real[i + threadIdx.y][threadIdx.x] =  cuCrealf(A[i * lda]);
            a_imag[i + threadIdx.y][threadIdx.x] = -cuCimagf(A[i * lda]);
          }
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
            a_imag[i + threadIdx.y][threadIdx.x] = cuCimagf(A[i * lda]);
          }
        }
        A += kb;
      }

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
            caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                  (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                                              (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
          else if (ti < l)
            caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                  (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
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
          caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                                            (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        else if (ti < l)
          caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for CGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? m - bi - mb : bi)
                                  : ((uplo == CBlasUpper) ? bi : m - bi - mb);
  while (k > 0) {
    if (trans != CBlasNoTrans) {
      if (trans == CBlasConjTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real[i + threadIdx.y][threadIdx.x] =  cuCrealf(A[i * lda]);
          a_imag[i + threadIdx.y][threadIdx.x] = -cuCimagf(A[i * lda]);
        }
      }
      else {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
          a_imag[i + threadIdx.y][threadIdx.x] = cuCimagf(A[i * lda]);
        }
      }
      A += kb;
    }

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
      b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
    }

    __syncthreads();

    if (k < kb) break;

    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l++) {
        caxpy(A[0], b_real[l], b_imag[l], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l++)
        caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]), &b_real[l][tj], &b_imag[l][tj], x);
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      caxpy(A[0], b_real[l], b_imag[l], x);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]), &b_real[l][tj], &b_imag[l][tj], x);
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {

    __syncthreads();

    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
        if (trans == CBlasConjTrans) {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real[i + threadIdx.y][threadIdx.x] =  cuCrealf(A[i * lda]);
            a_imag[i + threadIdx.y][threadIdx.x] = -cuCimagf(A[i * lda]);
          }
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real[i + threadIdx.y][threadIdx.x] = cuCrealf(A[i * lda]);
            a_imag[i + threadIdx.y][threadIdx.x] = cuCimagf(A[i * lda]);
          }
        }
        A += kb;
      }

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
            caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                  (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                                              (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
          else if (ti > l)
            caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                  (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
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
          caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                                            (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        else if (ti > l)
          caxpy((trans == CBlasNoTrans) ? A[0] : make_cuComplex(a_real[ti][ll], a_imag[ti][ll]),
                (trans == CBlasNoTrans) ? b_real[ll] : &b_real[ll][tj],
                (trans == CBlasNoTrans) ? b_imag[ll] : &b_imag[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }
  }

  n -= bj + tj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmulf(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 3]); if ( 4 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 4]); if ( 5 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 5]); if ( 6 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 6]); if ( 7 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 7]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmm2R(int m, int n,
                        cuComplex alpha, const cuComplex * __restrict__ A, int lda, const cuComplex * __restrict__ B, int ldb,
                        cuComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + threadIdx.x
                              : (bj + threadIdx.y) * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bi + ti : bj * ldb + bi + ti;
  }
  else {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + bj + threadIdx.x
                              : threadIdx.y * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bj * ldb + bi + ti : bi + ti;
  }
  X += bj * ldx + bi + ti;

  __shared__ float a_real[kb][nb];
  __shared__ float a_imag[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {
    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by) {
          a_real[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[j * lda]);
          a_imag[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? 0.0f
                                                                     : cuCimagf(A[j * lda]);
        }
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[l * lda]);
          a_imag[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 0.0f
                                                                     : -cuCimagf(A[l * lda]);
        }
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[l * lda]);
          a_imag[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 0.0f
                                                                     : cuCimagf(A[l * lda]);
        }
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         caxpy(ll + 1, B[0], a_real[ll], a_imag[ll], x);
        caxpy( 1, B[0], a_real[ 0], a_imag[ 0], x); B += ldb;
        caxpy( 2, B[0], a_real[ 1], a_imag[ 1], x); B += ldb;
        caxpy( 3, B[0], a_real[ 2], a_imag[ 2], x); B += ldb;
        caxpy( 4, B[0], a_real[ 3], a_imag[ 3], x); B += ldb;
        caxpy( 5, B[0], a_real[ 4], a_imag[ 4], x); B += ldb;
        caxpy( 6, B[0], a_real[ 5], a_imag[ 5], x); B += ldb;
        caxpy( 7, B[0], a_real[ 6], a_imag[ 6], x); B += ldb;
        caxpy( 8, B[0], a_real[ 7], a_imag[ 7], x); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    for (int ll = 0; ll < k; ll++) {
      caxpy(ll + 1, B[0], a_real[ll], a_imag[ll], x);
      B += ldb;
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for CGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? bj : n - bj - nb)
                                  : ((uplo == CBlasUpper) ? n - bj - nb : bj);
  while (k > 0) {
    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        a_real[threadIdx.x][j + threadIdx.y] = cuCrealf(A[j * lda]);
        a_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(A[j * lda]);
      }
    }
    else if (trans == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
        a_real[l + threadIdx.y][threadIdx.x] =  cuCrealf(A[l * lda]);
        a_imag[l + threadIdx.y][threadIdx.x] = -cuCimagf(A[l * lda]);
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
        a_real[l + threadIdx.y][threadIdx.x] = cuCrealf(A[l * lda]);
        a_imag[l + threadIdx.y][threadIdx.x] = cuCimagf(A[l * lda]);
      }
    }

      __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      caxpy(B[0], a_real[l], a_imag[l], x);
      B += ldb;
    }

    __syncthreads();

    A += (trans == CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a_real[l], a_imag[l], x);
    B += ldb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {

    __syncthreads();

    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by) {
          a_real[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[j * lda]);
          a_imag[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? 0.0f
                                                                     : cuCimagf(A[j * lda]);
        }
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[l * lda]);
          a_imag[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 0.0f
                                                                     : -cuCimagf(A[l * lda]);
        }
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 1.0f
                                                                     : cuCrealf(A[l * lda]);
          a_imag[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? 0.0f
                                                                     : cuCimagf(A[l * lda]);
        }
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         caxpy(nb - ll, B[0], &a_real[ll][ll], &a_imag[ll][ll], &x[ll]);
        caxpy(8, B[0], &a_real[ 0][ 0], &a_imag[ 0][ 0], &x[ 0]); B += ldb;
        caxpy(7, B[0], &a_real[ 1][ 1], &a_imag[ 1][ 1], &x[ 1]); B += ldb;
        caxpy(6, B[0], &a_real[ 2][ 2], &a_imag[ 2][ 2], &x[ 2]); B += ldb;
        caxpy(5, B[0], &a_real[ 3][ 3], &a_imag[ 3][ 3], &x[ 3]); B += ldb;
        caxpy(4, B[0], &a_real[ 4][ 4], &a_imag[ 4][ 4], &x[ 4]); B += ldb;
        caxpy(3, B[0], &a_real[ 5][ 5], &a_imag[ 5][ 5], &x[ 5]); B += ldb;
        caxpy(2, B[0], &a_real[ 6][ 6], &a_imag[ 6][ 6], &x[ 6]); B += ldb;
        caxpy(1, B[0], &a_real[ 7][ 7], &a_imag[ 7][ 7], &x[ 7]); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

//     for (int ll = 0; ll < k; ll++) {
//       caxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
//       B += ldb;
//     }
    if (k > 0) { caxpy(8, B[0], &a_real[ 0][ 0], &a_imag[ 0][ 0], &x[ 0]); B += ldb;
    if (k > 1) { caxpy(7, B[0], &a_real[ 1][ 1], &a_imag[ 1][ 1], &x[ 1]); B += ldb;
    if (k > 2) { caxpy(6, B[0], &a_real[ 2][ 2], &a_imag[ 2][ 2], &x[ 2]); B += ldb;
    if (k > 3) { caxpy(5, B[0], &a_real[ 3][ 3], &a_imag[ 3][ 3], &x[ 3]); B += ldb;
    if (k > 4) { caxpy(4, B[0], &a_real[ 4][ 4], &a_imag[ 4][ 4], &x[ 4]); B += ldb;
    if (k > 5) { caxpy(3, B[0], &a_real[ 5][ 5], &a_imag[ 5][ 5], &x[ 5]); B += ldb;
    if (k > 6) { caxpy(2, B[0], &a_real[ 6][ 6], &a_imag[ 6][ 6], &x[ 6]); B += ldb;
    if (k > 7) { caxpy(1, B[0], &a_real[ 7][ 7], &a_imag[ 7][ 7], &x[ 7]); B += ldb; }}}}}}}}
  }

  n -= bj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmulf(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 3]); if ( 4 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 4]); if ( 5 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 5]); if ( 6 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 6]); if ( 7 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 7]);
}

#else

// y(1:8) += alpha * x(1:8)
__device__ void caxpy(cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCfmaf(alpha, x[0], y[0]);
  y[1] = cuCfmaf(alpha, x[1], y[1]);
  y[2] = cuCfmaf(alpha, x[2], y[2]);
  y[3] = cuCfmaf(alpha, x[3], y[3]);
  y[4] = cuCfmaf(alpha, x[4], y[4]);
  y[5] = cuCfmaf(alpha, x[5], y[5]);
  y[6] = cuCfmaf(alpha, x[6], y[6]);
  y[7] = cuCfmaf(alpha, x[7], y[7]);
}

// y(1:n) += alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCfmaf(alpha, x[0], y[0]); if (1 >= n) return;
  y[1] = cuCfmaf(alpha, x[1], y[1]); if (2 >= n) return;
  y[2] = cuCfmaf(alpha, x[2], y[2]); if (3 >= n) return;
  y[3] = cuCfmaf(alpha, x[3], y[3]); if (4 >= n) return;
  y[4] = cuCfmaf(alpha, x[4], y[4]); if (5 >= n) return;
  y[5] = cuCfmaf(alpha, x[5], y[5]); if (6 >= n) return;
  y[6] = cuCfmaf(alpha, x[6], y[6]); if (7 >= n) return;
  y[7] = cuCfmaf(alpha, x[7], y[7]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmm2L(int m, int n,
                        cuComplex alpha, const cuComplex * __restrict__ A, int lda, const cuComplex * __restrict__ B, int ldb,
                        cuComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (trans != CBlasNoTrans) {
    tj = 8 * (ti / mb);
    ti = ti % mb;
  }

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? bi * lda + bi + ti : bi + ti;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + bi + threadIdx.x
                              : (bj + threadIdx.y) * ldb + threadIdx.x;
  }
  else {
    A += (uplo == CBlasUpper) ? (bi + threadIdx.y) * lda + threadIdx.x
                              : (bi + threadIdx.y) * lda + bi + threadIdx.x;
    B += (uplo == CBlasUpper) ? (bj + threadIdx.y) * ldb + threadIdx.x
                              : (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  }
  X += (bj + tj) * ldx + bi + ti;

  __shared__ cuComplex a[mb][kb + 1];
  __shared__ cuComplex b[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {
    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
        if (trans == CBlasConjTrans) {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][threadIdx.x] = cuConjf(A[i * lda]);
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][threadIdx.x] = A[i * lda];
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
            caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti < l)
            caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
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
          caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti < l)
          caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for CGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? m - bi - mb : bi)
                                  : ((uplo == CBlasUpper) ? bi : m - bi - mb);
  while (k > 0) {
    if (trans != CBlasNoTrans) {
      if (trans == CBlasConjTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][threadIdx.x] = cuConjf(A[i * lda]);
      }
      else {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][threadIdx.x] = A[i * lda];
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
        caxpy(A[0], b[l], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l++)
        caxpy(a[ti][l], &b[l][tj], x);
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      caxpy(A[0], b[l], x);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      caxpy(a[ti][l], &b[l][tj], x);
  }

  // For Upper/Trans and Lower/NoTrans process diagonal last
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {

    __syncthreads();

    int k = min(m - bi, mb);
    int l = 0;
    while (k > 0) {
      if (trans != CBlasNoTrans) {
        if (trans == CBlasConjTrans) {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][threadIdx.x] = cuConjf(A[i * lda]);
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][threadIdx.x] = A[i * lda];
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
          if (ti >= l++)
            caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti > l)
            caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
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
          caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          caxpy(make_cuComplex(1.0f, 0.0f), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti > l)
          caxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }
  }

  n -= bj + tj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmulf(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 3]); if ( 4 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 4]); if ( 5 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 5]); if ( 6 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 6]); if ( 7 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 7]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ctrmm2R(int m, int n,
                        cuComplex alpha, const cuComplex * __restrict__ A, int lda, const cuComplex * __restrict__ B, int ldb,
                        cuComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  if (trans == CBlasNoTrans) {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + threadIdx.x
                              : (bj + threadIdx.y) * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bi + ti : bj * ldb + bi + ti;
  }
  else {
    A += (uplo == CBlasUpper) ? (bj + threadIdx.y) * lda + bj + threadIdx.x
                              : threadIdx.y * lda + bj + threadIdx.x;
    B += (uplo == CBlasUpper) ? bj * ldb + bi + ti : bi + ti;
  }
  X += bj * ldx + bi + ti;

  __shared__ cuComplex a[kb][nb];

  cuComplex x[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {
    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          a[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : A[j * lda];
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : cuConjf(A[l * lda]);
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         caxpy(ll + 1, B[0], a[ll], x);
        caxpy( 1, B[0], a[ 0], x); B += ldb;
        caxpy( 2, B[0], a[ 1], x); B += ldb;
        caxpy( 3, B[0], a[ 2], x); B += ldb;
        caxpy( 4, B[0], a[ 3], x); B += ldb;
        caxpy( 5, B[0], a[ 4], x); B += ldb;
        caxpy( 6, B[0], a[ 5], x); B += ldb;
        caxpy( 7, B[0], a[ 6], x); B += ldb;
        caxpy( 8, B[0], a[ 7], x); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    for (int ll = 0; ll < k; ll++) {
      caxpy(ll + 1, B[0], a[ll], x);
      B += ldb;
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for CGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? bj : n - bj - nb)
                                  : ((uplo == CBlasUpper) ? n - bj - nb : bj);
  while (k > 0) {
    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by)
        a[threadIdx.x][j + threadIdx.y] = A[j * lda];
    }
    else if (trans == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by)
        a[l + threadIdx.y][threadIdx.x] = cuConjf(A[l * lda]);
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
      caxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += (trans == CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    caxpy(B[0], a[l], x);
    B += ldb;
  }

  // For Upper/NoTrans and Lower/Trans process diagonal last
  if (uplo == CBlasUpper && trans == CBlasNoTrans ||
      uplo == CBlasLower && trans != CBlasNoTrans) {

    __syncthreads();

    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          a[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : A[j * lda];
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : cuConjf(A[l * lda]);
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuComplex(1.0f, 0.0f) : A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         caxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
        caxpy(8, B[0], &a[ 0][ 0], &x[ 0]); B += ldb;
        caxpy(7, B[0], &a[ 1][ 1], &x[ 1]); B += ldb;
        caxpy(6, B[0], &a[ 2][ 2], &x[ 2]); B += ldb;
        caxpy(5, B[0], &a[ 3][ 3], &x[ 3]); B += ldb;
        caxpy(4, B[0], &a[ 4][ 4], &x[ 4]); B += ldb;
        caxpy(3, B[0], &a[ 5][ 5], &x[ 5]); B += ldb;
        caxpy(2, B[0], &a[ 6][ 6], &x[ 6]); B += ldb;
        caxpy(1, B[0], &a[ 7][ 7], &x[ 7]); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

//     for (int ll = 0; ll < k; ll++) {
//       caxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
//       B += ldb;
//     }
    if (k > 0) { caxpy(8, B[0], &a[ 0][ 0], &x[ 0]); B += ldb;
    if (k > 1) { caxpy(7, B[0], &a[ 1][ 1], &x[ 1]); B += ldb;
    if (k > 2) { caxpy(6, B[0], &a[ 2][ 2], &x[ 2]); B += ldb;
    if (k > 3) { caxpy(5, B[0], &a[ 3][ 3], &x[ 3]); B += ldb;
    if (k > 4) { caxpy(4, B[0], &a[ 4][ 4], &x[ 4]); B += ldb;
    if (k > 5) { caxpy(3, B[0], &a[ 5][ 5], &x[ 5]); B += ldb;
    if (k > 6) { caxpy(2, B[0], &a[ 6][ 6], &x[ 6]); B += ldb;
    if (k > 7) { caxpy(1, B[0], &a[ 7][ 7], &x[ 7]); B += ldb; }}}}}}}}
  }

  n -= bj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmulf(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 3]); if ( 4 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 4]); if ( 5 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 5]); if ( 6 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 6]); if ( 7 >= n) return; X += ldx;
  X[0] = cuCmulf(alpha, x[ 7]);
}

#endif

template void ctrmm2L<CBlasUpper, CBlasNoTrans,     CBlasUnit,    64,  8, 16, 16,  4>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasUpper, CBlasNoTrans,     CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasUpper, CBlasTrans,       CBlasUnit,    32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasUpper, CBlasTrans,       CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasUpper, CBlasConjTrans,   CBlasUnit,    32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasUpper, CBlasConjTrans,   CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasNoTrans,     CBlasUnit,    64,  8, 16, 16,  4>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasNoTrans,     CBlasNonUnit, 64,  8, 16, 16,  4>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasTrans,       CBlasUnit,    32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasTrans,       CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasConjTrans,   CBlasUnit,    32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2L<CBlasLower, CBlasConjTrans,   CBlasNonUnit, 32, 16,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);

template void ctrmm2R<CBlasUpper, CBlasNoTrans,     CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasUpper, CBlasNoTrans,     CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasUpper, CBlasTrans,       CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasUpper, CBlasTrans,       CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasUpper, CBlasConjTrans,   CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasUpper, CBlasConjTrans,   CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasNoTrans,     CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasNoTrans,     CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasTrans,       CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasTrans,       CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasConjTrans,   CBlasUnit,    64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrmm2R<CBlasLower, CBlasConjTrans,   CBlasNonUnit, 64,  8,  8,  8,  8>(int, int, cuComplex, const cuComplex * __restrict__, int, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
