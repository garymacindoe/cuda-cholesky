#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICT__)

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy(cuDoubleComplex alpha, const int * x_real_hi, const int * x_real_lo,
                      const int * x_imag_hi, const int * x_imag_lo, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                            __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);
  y[1] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                            __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);
  y[2] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[2], x_real_lo[2]),
                                            __hiloint2double(x_imag_hi[2], x_imag_lo[2])), y[2]);
  y[3] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[3], x_real_lo[3]),
                                            __hiloint2double(x_imag_hi[3], x_imag_lo[3])), y[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex alpha, const int * x_real_hi, const int * x_real_lo,
                      const int * x_imag_hi, const int * x_imag_lo, cuDoubleComplex * y) {
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

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmm2L(int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ A, int lda, const cuDoubleComplex * __restrict__ B, int ldb,
                        cuDoubleComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (trans != CBlasNoTrans) {
    tj = 4 * (ti / mb);
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

  __shared__ int a_real_hi[mb][kb + 1];
  __shared__ int a_real_lo[mb][kb + 1];
  __shared__ int a_imag_hi[mb][kb + 1];
  __shared__ int a_imag_lo[mb][kb + 1];
  __shared__ int b_real_hi[kb][nb];
  __shared__ int b_real_lo[kb][nb];
  __shared__ int b_imag_hi[kb][nb];
  __shared__ int b_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

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
            a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint( cuCreal(A[i * lda]));
            a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint( cuCreal(A[i * lda]));
            a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(-cuCimag(A[i * lda]));
            a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(-cuCimag(A[i * lda]));
          }
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
            a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
            a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[i * lda]));
            a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[i * lda]));
          }
        }
        A += kb;
      }

#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
        b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
        b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
        b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
      }

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti <= l++)
            zaxpy((trans == CBlasNoTrans) ? A[0]
                                          : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                                 __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            zaxpy(make_cuDoubleComplex(1.0, 0.0),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
          else if (ti < l)
            zaxpy((trans == CBlasNoTrans) ? A[0]
                                          : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                                 __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
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
          zaxpy((trans == CBlasNoTrans) ? A[0]
                                        : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                               __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          zaxpy(make_cuDoubleComplex(1.0, 0.0),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        else if (ti < l)
          zaxpy((trans == CBlasNoTrans) ? A[0]
                                        : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                               __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for ZGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? m - bi - mb : bi)
                                  : ((uplo == CBlasUpper) ? bi : m - bi - mb);
  while (k > 0) {
    if (trans != CBlasNoTrans) {
      if (trans == CBlasConjTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint( cuCreal(A[i * lda]));
          a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint( cuCreal(A[i * lda]));
          a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(-cuCimag(A[i * lda]));
          a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(-cuCimag(A[i * lda]));
        }
      }
      else {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
          a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
          a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[i * lda]));
          a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[i * lda]));
        }
      }
      A += kb;
    }

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
      b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
      b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
      b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
    }

    __syncthreads();

    if (k < kb) break;

    if (trans == CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l++) {
        zaxpy(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l++)
        zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                                   __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
              &b_real_hi[l][tj], &b_real_lo[l][tj], &b_imag_hi[l][tj], &b_imag_lo[l][tj], x);
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      zaxpy(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], x);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                                 __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
            &b_real_hi[l][tj], &b_real_lo[l][tj], &b_imag_hi[l][tj], &b_imag_lo[l][tj], x);
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
            a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint( cuCreal(A[i * lda]));
            a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint( cuCreal(A[i * lda]));
            a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(-cuCimag(A[i * lda]));
            a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(-cuCimag(A[i * lda]));
          }
        }
        else {
#pragma unroll
          for (int i = 0; i < mb; i += by) {
            a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
            a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
            a_imag_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[i * lda]));
            a_imag_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[i * lda]));
          }
        }
        A += kb;
      }

#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
        b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
        b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
        b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
      }

      __syncthreads();

      if (k < kb) break;

      if (diag == CBlasNonUnit) {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti <= l++)
            zaxpy((trans == CBlasNoTrans) ? A[0]
                                          : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                                 __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            zaxpy(make_cuDoubleComplex(1.0, 0.0),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
          else if (ti < l)
            zaxpy((trans == CBlasNoTrans) ? A[0]
                                          : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                                 __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                  (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                  (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
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
          zaxpy((trans == CBlasNoTrans) ? A[0]
                                        : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                               __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          zaxpy(make_cuDoubleComplex(1.0, 0.0),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        else if (ti < l)
          zaxpy((trans == CBlasNoTrans) ? A[0]
                                        : make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                                               __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                (trans == CBlasNoTrans) ? b_real_hi[ll] : &b_real_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_real_lo[ll] : &b_real_lo[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_hi[ll] : &b_imag_hi[ll][tj],
                (trans == CBlasNoTrans) ? b_imag_lo[ll] : &b_imag_lo[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }
  }

  n -= bj + tj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmul(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 3]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmm2R(int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ A, int lda, const cuDoubleComplex * __restrict__ B, int ldb,
                        cuDoubleComplex * __restrict__ X, int ldx) {

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

  __shared__ int a_real_hi[kb][nb];
  __shared__ int a_real_lo[kb][nb];
  __shared__ int a_imag_hi[kb][nb];
  __shared__ int a_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {
    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by) {
          a_real_hi[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[j * lda]));
          a_real_lo[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[j * lda]));
          a_imag_hi[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(cuCimag(A[j * lda]));
          a_imag_lo[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(cuCimag(A[j * lda]));
        }
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[l * lda]));
          a_real_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[l * lda]));
          a_imag_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(-cuCimag(A[l * lda]));
          a_imag_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(-cuCimag(A[l * lda]));
        }
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[l * lda]));
          a_real_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[l * lda]));
          a_imag_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(cuCimag(A[l * lda]));
          a_imag_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(cuCimag(A[l * lda]));
        }
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         zaxpy(ll + 1, B[0], a_real[ll], a_imag[ll], x);
        zaxpy(1, B[0], a_real_hi[0], a_real_lo[0], a_imag_lo[0], a_imag_lo[0], x); B += ldb;
        zaxpy(2, B[0], a_real_hi[1], a_real_lo[1], a_imag_lo[1], a_imag_lo[1], x); B += ldb;
        zaxpy(3, B[0], a_real_hi[2], a_real_lo[2], a_imag_lo[2], a_imag_lo[2], x); B += ldb;
        zaxpy(4, B[0], a_real_hi[3], a_real_lo[3], a_imag_lo[3], a_imag_lo[3], x); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    for (int ll = 0; ll < k; ll++) {
      zaxpy(ll + 1, B[0], a_real_hi[ll], a_real_lo[ll], a_imag_hi[ll], a_imag_lo[ll], x);
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
        a_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(A[j * lda]));
        a_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(A[j * lda]));
        a_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(A[j * lda]));
        a_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(A[j * lda]));
      }
    }
    else if (trans == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
        a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint( cuCreal(A[l * lda]));
        a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint( cuCreal(A[l * lda]));
        a_imag_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(-cuCimag(A[l * lda]));
        a_imag_lo[l + threadIdx.y][threadIdx.x] = __double2loint(-cuCimag(A[l * lda]));
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
        a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[l * lda]));
        a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[l * lda]));
        a_imag_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[l * lda]));
        a_imag_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[l * lda]));
      }
    }

      __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += (trans == CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
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
          a_real_hi[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[j * lda]));
          a_real_lo[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[j * lda]));
          a_imag_hi[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(cuCimag(A[j * lda]));
          a_imag_lo[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(cuCimag(A[j * lda]));
        }
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[l * lda]));
          a_real_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[l * lda]));
          a_imag_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(-cuCimag(A[l * lda]));
          a_imag_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(-cuCimag(A[l * lda]));
        }
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by) {
          a_real_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(1.0)
                                                                     : __double2hiint(cuCreal(A[l * lda]));
          a_real_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(1.0)
                                                                     : __double2loint(cuCreal(A[l * lda]));
          a_imag_hi[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2hiint(0.0)
                                                                     : __double2hiint(cuCimag(A[l * lda]));
          a_imag_lo[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? __double2loint(0.0)
                                                                     : __double2loint(cuCimag(A[l * lda]));
        }
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         zaxpy(nb - ll, B[0], &a_real[ll][ll], &a_imag[ll][ll], &x[ll]);
        zaxpy(4, B[0], &a_real_hi[0][0], &a_real_lo[0][0], &a_imag_hi[0][0], &a_imag_lo[0][0], &x[0]); B += ldb;
        zaxpy(3, B[0], &a_real_hi[1][1], &a_real_lo[1][1], &a_imag_hi[1][1], &a_imag_lo[1][1], &x[1]); B += ldb;
        zaxpy(2, B[0], &a_real_hi[2][2], &a_real_lo[2][2], &a_imag_hi[2][2], &a_imag_lo[2][2], &x[2]); B += ldb;
        zaxpy(1, B[0], &a_real_hi[3][3], &a_real_lo[3][3], &a_imag_hi[3][3], &a_imag_lo[3][3], &x[3]); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

//     for (int ll = 0; ll < k; ll++) {
//       zaxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
//       B += ldb;
//     }
    if (k > 0) { zaxpy(4, B[0], &a_real_hi[0][0], &a_real_lo[0][0], &a_imag_hi[0][0], &a_imag_lo[0][0], &x[0]); B += ldb;
    if (k > 1) { zaxpy(3, B[0], &a_real_hi[1][1], &a_real_lo[1][1], &a_imag_hi[1][1], &a_imag_lo[1][1], &x[1]); B += ldb;
    if (k > 2) { zaxpy(2, B[0], &a_real_hi[2][2], &a_real_lo[2][2], &a_imag_hi[2][2], &a_imag_lo[2][2], &x[2]); B += ldb;
    if (k > 3) { zaxpy(1, B[0], &a_real_hi[3][3], &a_real_lo[3][3], &a_imag_hi[3][3], &a_imag_lo[3][3], &x[3]); B += ldb; }}}}
  }

  n -= bj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmul(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 3]);
}

#else

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy(cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, x[0], y[0]);
  y[1] = cuCfma(alpha, x[1], y[1]);
  y[2] = cuCfma(alpha, x[2], y[2]);
  y[3] = cuCfma(alpha, x[3], y[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, x[0], y[0]); if (1 >= n) return;
  y[1] = cuCfma(alpha, x[1], y[1]); if (2 >= n) return;
  y[2] = cuCfma(alpha, x[2], y[2]); if (3 >= n) return;
  y[3] = cuCfma(alpha, x[3], y[3]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmm2L(int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ A, int lda, const cuDoubleComplex * __restrict__ B, int ldb,
                        cuDoubleComplex * __restrict__ X, int ldx) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (trans != CBlasNoTrans) {
    tj = 4 * (ti / mb);
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

  __shared__ cuDoubleComplex a[mb][kb + 1];
  __shared__ cuDoubleComplex b[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

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
            a[i + threadIdx.y][threadIdx.x] = cuConj(A[i * lda]);
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
            zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            zaxpy(make_cuDoubleComplex(1.0, 0.0), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti < l)
            zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
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
          zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          zaxpy(make_cuDoubleComplex(1.0, 0.0), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti < l)
          zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
        l++;
      }
    }

    __syncthreads();
  }

  // Process non-diagonal blocks as for ZGEMM
  int k = (trans == CBlasNoTrans) ? ((uplo == CBlasUpper) ? m - bi - mb : bi)
                                  : ((uplo == CBlasUpper) ? bi : m - bi - mb);
  while (k > 0) {
    if (trans != CBlasNoTrans) {
      if (trans == CBlasConjTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][threadIdx.x] = cuConj(A[i * lda]);
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
        zaxpy(A[0], b[l], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l++)
        zaxpy(a[ti][l], &b[l][tj], x);
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      zaxpy(A[0], b[l], x);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      zaxpy(a[ti][l], &b[l][tj], x);
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
            a[i + threadIdx.y][threadIdx.x] = cuConj(A[i * lda]);
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
            zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                  (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          if (trans == CBlasNoTrans)
            A += lda;
        }
      }
      else {
#pragma unroll
        for (int ll = 0; ll < kb; ll++) {
          if (ti == l)
            zaxpy(make_cuDoubleComplex(1.0, 0.0), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
          else if (ti > l)
            zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
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
          zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
                (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        if (trans == CBlasNoTrans)
          A += lda;
      }
    }
    else {
      for (int ll = 0; ll < k; ll++) {
        if (ti == l)
          zaxpy(make_cuDoubleComplex(1.0, 0.0), (trans == CBlasNoTrans) ? b[ll] : &b[ll][tj], x);
        else if (ti > l)
          zaxpy((trans == CBlasNoTrans) ? A[0]  :  a[ti][ll],
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
  X[0] = cuCmul(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 3]);
}

template <CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmm2R(int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ A, int lda, const cuDoubleComplex * __restrict__ B, int ldb,
                        cuDoubleComplex * __restrict__ X, int ldx) {

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

  __shared__ cuDoubleComplex a[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  if (uplo == CBlasUpper && trans != CBlasNoTrans ||
      uplo == CBlasLower && trans == CBlasNoTrans) {
    int k = min(n - bj, nb);
    while (k > 0) {
      if (trans == CBlasNoTrans) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          a[threadIdx.x][j + threadIdx.y] =
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : A[j * lda];
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : cuConj(A[l * lda]);
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         zaxpy(ll + 1, B[0], a[ll], x);
        zaxpy( 1, B[0], a[ 0], x); B += ldb;
        zaxpy( 2, B[0], a[ 1], x); B += ldb;
        zaxpy( 3, B[0], a[ 2], x); B += ldb;
        zaxpy( 4, B[0], a[ 3], x); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

    for (int ll = 0; ll < k; ll++) {
      zaxpy(ll + 1, B[0], a[ll], x);
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
        a[l + threadIdx.y][threadIdx.x] = cuConj(A[l * lda]);
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
      zaxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += (trans == CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a[l], x);
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
            (diag != CBlasNonUnit && threadIdx.x == j + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : A[j * lda];
      }
      else if (trans == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : cuConj(A[l * lda]);
      }
      else {
#pragma unroll
        for (int l = 0; l < kb; l += by)
          a[l + threadIdx.y][threadIdx.x] =
            (diag != CBlasNonUnit && threadIdx.x == l + threadIdx.y) ? make_cuDoubleComplex(1.0, 0.0) : A[l * lda];
      }

      __syncthreads();

      if (k < kb) break;

// #pragma unroll
//       for (int ll = 0; ll < kb; ll++) {
//         zaxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
        zaxpy(4, B[0], &a[0][0], &x[0]); B += ldb;
        zaxpy(3, B[0], &a[1][1], &x[1]); B += ldb;
        zaxpy(2, B[0], &a[2][2], &x[2]); B += ldb;
        zaxpy(1, B[0], &a[3][3], &x[3]); B += ldb;
//         B += ldb;
//       }

      __syncthreads();

      A += (trans == CBlasNoTrans) ? kb : kb * lda;
      k -= kb;
    }

//     for (int ll = 0; ll < k; ll++) {
//       zaxpy(nb - ll, B[0], &a[ll][ll], &x[ll]);
//       B += ldb;
//     }
    if (k > 0) { zaxpy(4, B[0], &a[0][0], &x[0]); B += ldb;
    if (k > 1) { zaxpy(3, B[0], &a[1][1], &x[1]); B += ldb;
    if (k > 2) { zaxpy(2, B[0], &a[2][2], &x[2]); B += ldb;
    if (k > 3) { zaxpy(1, B[0], &a[3][3], &x[3]); B += ldb; }}}}
  }

  n -= bj;
  m -= bi + ti;
  if (n <= 0 || m <= 0) return;
  X[0] = cuCmul(alpha, x[ 0]); if ( 1 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 1]); if ( 2 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 2]); if ( 3 >= n) return; X += ldx;
  X[0] = cuCmul(alpha, x[ 3]);
}

#endif

template void ztrmm2L<CBlasUpper, CBlasNoTrans,     CBlasUnit,    64, 4, 16, 16,  4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasUpper, CBlasNoTrans,     CBlasNonUnit, 64, 4, 16, 16,  4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasUpper, CBlasTrans,       CBlasUnit,    32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasUpper, CBlasTrans,       CBlasNonUnit, 32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasUpper, CBlasConjTrans,   CBlasUnit,    32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasUpper, CBlasConjTrans,   CBlasNonUnit, 32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasNoTrans,     CBlasUnit,    64, 4, 16, 16,  4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasNoTrans,     CBlasNonUnit, 64, 4, 16, 16,  4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasTrans,       CBlasUnit,    32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasTrans,       CBlasNonUnit, 32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasConjTrans,   CBlasUnit,    32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2L<CBlasLower, CBlasConjTrans,   CBlasNonUnit, 32, 8,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);

template void ztrmm2R<CBlasUpper, CBlasNoTrans,     CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasUpper, CBlasNoTrans,     CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasUpper, CBlasTrans,       CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasUpper, CBlasTrans,       CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasUpper, CBlasConjTrans,   CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasUpper, CBlasConjTrans,   CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasNoTrans,     CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasNoTrans,     CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasTrans,       CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasTrans,       CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasConjTrans,   CBlasUnit,    64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrmm2R<CBlasLower, CBlasConjTrans,   CBlasNonUnit, 64, 4,  8,  8,  8>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
