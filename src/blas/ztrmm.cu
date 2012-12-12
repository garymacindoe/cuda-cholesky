#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICTS__)

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy(cuDoubleComplex alpha, const int * __restrict__ x_real_hi,
                      const int * __restrict__ x_real_lo,
                      const int * __restrict__ x_imag_hi,
                      const int * __restrict__ x_imag_lo,
                      cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                            __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);
  y[1] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                            __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);
  y[2] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[2], x_real_lo[2]),
                                            __hiloint2double(x_imag_hi[2], x_imag_lo[2])), y[2]);
  y[3] = cuCfma(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[3], x_real_lo[3]),
                                            __hiloint2double(x_imag_hi[3], x_imag_lo[3])), y[3]);
}

// y(1:4) += x(1:4)
__device__ void zaxpy(const int * __restrict__ x_real_hi,
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
__device__ void zaxpy(int n, cuDoubleComplex alpha,
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

// y(1:n) = alpha * x(1:n)
__device__ void zscal(int n, cuDoubleComplex alpha,
                      const cuDoubleComplex * __restrict__ x,
                      cuDoubleComplex * __restrict__ y, int incy) {
  if (n <= 0) return;
  y[0] = cuCmul(alpha, x[0]); if (1 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[1]); if (2 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[2]); if (3 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[3]);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLUN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B,
                         cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ int b_real_hi[kb][nb];
  __shared__ int b_real_lo[kb][nb];
  __shared__ int b_imag_hi[kb][nb];
  __shared__ int b_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
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
          zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
        else if (ti < l)
          zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
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
        zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      else if (ti < l)
        zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for ZGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
      b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
      b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
      b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLUT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 4 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
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

  // Process non-diagonal blocks as for ZGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
      a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
      a_imag_hi[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[i * lda]))
                                                : __double2hiint( cuCimag(A[i * lda]));
      a_imag_lo[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[i * lda]))
                                                : __double2loint( cuCimag(A[i * lda]));
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
      b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
      b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
      b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                                 __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
            &b_real_hi[l][tj], &b_real_lo[l][tj], &b_imag_hi[l][tj], &b_imag_lo[l][tj], x);

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
      a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
      a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
      a_imag_hi[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[i * lda]))
                                                : __double2hiint( cuCimag(A[i * lda]));
      a_imag_lo[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[i * lda]))
                                                : __double2loint( cuCimag(A[i * lda]));
    }
    A += kb;

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
        if (ti >= l++)
          zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                     __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(&b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
        else if (ti > l)
          zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                     __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
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
        zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                   __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
              &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(&b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      else if (ti > l)
        zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                   __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
              &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    zscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLLN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ int b_real_hi[kb][nb];
  __shared__ int b_real_lo[kb][nb];
  __shared__ int b_imag_hi[kb][nb];
  __shared__ int b_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
      b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
      b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
      b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], x);
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
        if (ti >= l++)
          zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
        else if (ti > l)
          zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
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
        zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      else if (ti > l)
        zaxpy(A[0], b_real_hi[ll], b_real_lo[ll], b_imag_hi[ll], b_imag_lo[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLLT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 4 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + bi + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
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
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
      a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
      a_imag_hi[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[i * lda]))
                                                : __double2hiint( cuCimag(A[i * lda]));
      a_imag_lo[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[i * lda]))
                                                : __double2loint( cuCimag(A[i * lda]));
    }
    A += kb;

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
          zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                     __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(&b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
        else if (ti < l)
          zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                     __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
                &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
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
        zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                   __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
              &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(&b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      else if (ti < l)
        zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][ll], a_real_lo[ti][ll]),
                                   __hiloint2double(a_imag_hi[ti][ll], a_imag_lo[ti][ll])),
              &b_real_hi[ll][tj], &b_real_lo[ll][tj], &b_imag_hi[ll][tj], &b_imag_lo[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for ZGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by) {
      a_real_hi[i + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[i * lda]));
      a_real_lo[i + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[i * lda]));
      a_imag_hi[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[i * lda]))
                                                : __double2hiint( cuCimag(A[i * lda]));
      a_imag_lo[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[i * lda]))
                                                : __double2loint( cuCimag(A[i * lda]));
    }
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by) {
      b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
      b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
      b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
      b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                                 __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
            &b_real_hi[l][tj], &b_real_lo[l][tj], &b_imag_hi[l][tj], &b_imag_lo[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    zaxpy(make_cuDoubleComplex(__hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                               __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
          &b_real_hi[l][tj], &b_real_lo[l][tj], &b_imag_hi[l][tj], &b_imag_lo[l][tj], x);

  if (m - bi - ti > 0)
    zscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] = cuCadd(x[(i)], B[0]); \
      zaxpy(8 - (i) - 1, B[0], &a_real_hi[(i)][(i) + 1], &a_real_lo[(i)][(i) + 1], \
                               &a_imag_hi[(i)][(i) + 1], &a_imag_lo[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      zaxpy(8 - (i), B[0], &a_real_hi[(i)][(i)], &a_real_lo[(i)][(i)], \
                           &a_imag_hi[(i)][(i)], &a_imag_lo[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      zaxpy((i) - 1, B[0], a_real_hi[(i) - 1], a_real_lo[(i) - 1], \
                           a_imag_hi[(i) - 1], a_imag_lo[(i) - 1], x); \
      x[(i) - 1] = cuCadd(x[(i) - 1], B[0]); \
    } \
    else \
      zaxpy((i), B[0], a_real_hi[(i) - 1], a_real_lo[(i) - 1], a_imag_hi[(i) - 1], a_imag_lo[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRUN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_real_hi[kb][nb];
  __shared__ int a_real_lo[kb][nb];
  __shared__ int a_imag_hi[kb][nb];
  __shared__ int a_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(A[j * lda]));
      a_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(A[j * lda]));
      a_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(A[j * lda]));
      a_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(A[j * lda]));
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
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
      a_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(A[j * lda]));
      a_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(A[j * lda]));
      a_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(A[j * lda]));
      a_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(A[j * lda]));
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_DEC(3);

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRUT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_real_hi[kb][nb];
  __shared__ int a_real_lo[kb][nb];
  __shared__ int a_imag_hi[kb][nb];
  __shared__ int a_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[l * lda]));
      a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[l * lda]));
      a_imag_hi[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[l * lda]))
                                                : __double2hiint( cuCimag(A[l * lda]));
      a_imag_lo[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[l * lda]))
                                                : __double2loint( cuCimag(A[l * lda]));
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_INC(4);

  // Process non-diagonal blocks as for ZGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[l * lda]));
      a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[l * lda]));
      a_imag_hi[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[l * lda]))
                                                : __double2hiint( cuCimag(A[l * lda]));
      a_imag_lo[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[l * lda]))
                                                : __double2loint( cuCimag(A[l * lda]));
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRLN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_real_hi[kb][nb];
  __shared__ int a_real_lo[kb][nb];
  __shared__ int a_imag_hi[kb][nb];
  __shared__ int a_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(A[j * lda]));
      a_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(A[j * lda]));
      a_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(A[j * lda]));
      a_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(A[j * lda]));
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_INC(4);

  // Process non-diagonal blocks as for ZGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by) {
      a_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(A[j * lda]));
      a_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(A[j * lda]));
      a_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(A[j * lda]));
      a_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(A[j * lda]));
    }

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRLT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ int a_real_hi[kb][nb];
  __shared__ int a_real_lo[kb][nb];
  __shared__ int a_imag_hi[kb][nb];
  __shared__ int a_imag_lo[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by) {
      a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[l * lda]));
      a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[l * lda]));
      a_imag_hi[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[l * lda]))
                                                : __double2hiint( cuCimag(A[l * lda]));
      a_imag_lo[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[l * lda]))
                                                : __double2loint( cuCimag(A[l * lda]));
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);
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
      a_real_hi[l + threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[l * lda]));
      a_real_lo[l + threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[l * lda]));
      a_imag_hi[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2hiint(-cuCimag(A[l * lda]))
                                                : __double2hiint( cuCimag(A[l * lda]));
      a_imag_lo[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans)
                                                ? __double2loint(-cuCimag(A[l * lda]))
                                                : __double2loint( cuCimag(A[l * lda]));
    }

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_DEC(3);

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

#else

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy(cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, x[0], y[0]); y[1] = cuCfma(alpha, x[1], y[1]);
  y[2] = cuCfma(alpha, x[2], y[2]); y[3] = cuCfma(alpha, x[3], y[3]);
}

// y(1:4) += x(1:4)
__device__ void zaxpy(const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCadd(y[0], x[0]); y[1] = cuCadd(y[1], x[1]);
  y[2] = cuCadd(y[2], x[2]); y[3] = cuCadd(y[3], x[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  if (n <= 0) return;
  y[0] = cuCfma(alpha, x[0], y[0]); if (1 >= n) return; y[1] = cuCfma(alpha, x[1], y[1]); if (2 >= n) return;
  y[2] = cuCfma(alpha, x[2], y[2]); if (3 >= n) return; y[3] = cuCfma(alpha, x[3], y[3]);
}

// y(1:n) = alpha * x(1:n)
__device__ void zscal(int n, cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y, int incy) {
  if (n <= 0) return;
  y[0] = cuCmul(alpha, x[0]); if (1 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[1]); if (2 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[2]); if (3 >= n) return; y += incy;
  y[0] = cuCmul(alpha, x[3]);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLUN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi * lda + bi + ti;
  B += (bj + threadIdx.y) * ldb + bi + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex b[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

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
          zaxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(b[ll], x);
        else if (ti < l)
          zaxpy(A[0], b[ll], x);
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
        zaxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(b[ll], x);
      else if (ti < l)
        zaxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  // Process any non-diagonal blocks as for ZGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(A[0], b[l], x);
      A += lda;
    }

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(A[0], b[l], x);
    A += lda;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLUT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 4 * (ti / mb);
  ti = ti % mb;

  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += (bj + tj) * ldx + bi + ti;

  __shared__ cuDoubleComplex a[mb][kb + 1];
  __shared__ cuDoubleComplex b[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[i * lda]) : A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy(a[ti][l], &b[l][tj], x);

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
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[i * lda]) : A[i * lda];
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
          zaxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(&b[ll][tj], x);
        else if (ti > l)
          zaxpy(a[ti][ll], &b[ll][tj], x);
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
        zaxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(&b[ll][tj], x);
      else if (ti > l)
        zaxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  if (m - bi - ti > 0)
    zscal(n - bj - tj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLLN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += bi + ti;
  B += (bj + threadIdx.y) * ldb + threadIdx.x;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex b[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bi;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(A[0], b[l], x);
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
          zaxpy(A[0], b[ll], x);
        A += lda;
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(b[ll], x);
        else if (ti > l)
          zaxpy(A[0], b[ll], x);
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
        zaxpy(A[0], b[ll], x);
      A += lda;
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(b[ll], x);
      else if (ti > l)
        zaxpy(A[0], b[ll], x);
      A += lda;
      l++;
    }
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmLLT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
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

  __shared__ cuDoubleComplex a[mb][kb + 1];
  __shared__ cuDoubleComplex b[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/NoTrans and Lower/Trans process diagonal first
  int k = min(m - bi, mb);
  int l = 0;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[i * lda]) : A[i * lda];
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
          zaxpy(a[ti][ll], &b[ll][tj], x);
      }
    }
    else {
#pragma unroll
      for (int ll = 0; ll < kb; ll++) {
        if (ti == l)
          zaxpy(&b[ll][tj], x);
        else if (ti < l)
          zaxpy(a[ti][ll], &b[ll][tj], x);
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
        zaxpy(a[ti][ll], &b[ll][tj], x);
    }
  }
  else {
    for (int ll = 0; ll < k; ll++) {
      if (ti == l)
        zaxpy(&b[ll][tj], x);
      else if (ti < l)
        zaxpy(a[ti][ll], &b[ll][tj], x);
      l++;
    }
  }

  // Process any non-diagonal blocks as for ZGEMM
  k = m - bi - mb;
  while (k > 0) {
#pragma unroll
    for (int i = 0; i < mb; i += by)
      a[i + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[i * lda]) : A[i * lda];
    A += kb;

#pragma unroll
    for (int j = 0; j < nb; j += by)
      b[threadIdx.x][j + threadIdx.y] = B[j * ldb];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy(a[ti][l], &b[l][tj], x);

    __syncthreads();

    B += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    zaxpy(a[ti][l], &b[l][tj], x);

  if (m - bi - ti > 0)
    zscal(n - bj - tj, alpha, x, X, ldx);
}

#define INNER_RIGHT_LOOP_DEC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      x[(i)] = cuCadd(x[(i)], B[0]); \
      zaxpy(8 - (i) - 1, B[0], &a[(i)][(i) + 1], &x[(i) + 1]); \
    } \
    else \
      zaxpy(8 - (i), B[0], &a[(i)][(i)], &x[(i)]); \
  } while (false)

#define INNER_RIGHT_LOOP_INC(i) \
  do { \
    if (diag != CBlasNonUnit) { \
      zaxpy((i) - 1, B[0], a[(i) - 1], x); \
      x[(i) - 1] = cuCadd(x[(i) - 1], B[0]); \
    } \
    else \
      zaxpy((i), B[0], a[(i) - 1], x); \
  } while (false)

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRUN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex a[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a[l], x);
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

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_DEC(3);

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRUT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;


  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex a[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // For Upper/Trans and Lower/NoTrans process diagonal first
  int k = min(n - bj, nb);
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[l * lda]) : A[l * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_INC(1); B += ldb;
    INNER_RIGHT_LOOP_INC(2); B += ldb;
    INNER_RIGHT_LOOP_INC(3); B += ldb;
    INNER_RIGHT_LOOP_INC(4); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_INC(4);

  // Process non-diagonal blocks as for ZGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[l * lda]) : A[l * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRLN(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  int ti = threadIdx.y * bx + threadIdx.x;

  A += (bj + threadIdx.y) * lda + bj + threadIdx.x;
  B += bj * ldb + bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex a[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

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

    __syncthreads();

    A += kb;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_INC(1); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_INC(2); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_INC(3); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_INC(4);

  // Process non-diagonal blocks as for ZGEMM
  k = n - bj - nb;
  while (k > 0) {
#pragma unroll
    for (int j = 0; j < nb; j += by)
      a[threadIdx.x][j + threadIdx.y] = A[j * lda];

    __syncthreads();

    if (k < kb) break;

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a[l], x);
      B += ldb;
    }

    __syncthreads();

    A += kb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy(B[0], a[l], x);
    B += ldb;
  }

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

template <CBlasTranspose trans, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ztrmmRLT(const cuDoubleComplex * __restrict__ A,
                         const cuDoubleComplex * __restrict__ B, cuDoubleComplex * __restrict__ X,
                         cuDoubleComplex alpha,
                         int lda, int ldb, int ldx,
                         int m, int n) {

  const int bi = blockIdx.x * mb;       // Starting row of block of X
  const int bj = blockIdx.y * nb;       // Starting column of block of X
  const int ti = threadIdx.y * bx + threadIdx.x;

  A += threadIdx.y * lda + bj + threadIdx.x;
  B += bi + ti;
  X += bj * ldx + bi + ti;

  __shared__ cuDoubleComplex a[kb][nb];

  cuDoubleComplex x[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  // Process non-diagonal blocks as for ZGEMM
  int k = bj;
  while (k > 0) {
#pragma unroll
    for (int l = 0; l < kb; l += by)
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[l * lda]) : A[l * lda];

    __syncthreads();

#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy(B[0], a[l], x);
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
      a[l + threadIdx.y][threadIdx.x] = (trans == CBlasConjTrans) ? cuConj(A[l * lda]) : A[l * lda];

    __syncthreads();

    if (k < kb) break;

    INNER_RIGHT_LOOP_DEC(0); B += ldb; INNER_RIGHT_LOOP_DEC(1); B += ldb;
    INNER_RIGHT_LOOP_DEC(2); B += ldb; INNER_RIGHT_LOOP_DEC(3); B += ldb;

    __syncthreads();

    A += kb * lda;
    k -= kb;
  }

  if (k > 0) { INNER_RIGHT_LOOP_DEC(0); B += ldb; }
  if (k > 1) { INNER_RIGHT_LOOP_DEC(1); B += ldb; }
  if (k > 2) { INNER_RIGHT_LOOP_DEC(2); B += ldb; }
  if (k > 3)   INNER_RIGHT_LOOP_DEC(3);

  if (m - bi - ti > 0)
    zscal(n - bj, alpha, x, X, ldx);
}

#endif

template void ztrmmLUN<CBlasUnit,    64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLUN<CBlasNonUnit, 64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLUT<CBlasTrans,     CBlasUnit,    8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLUT<CBlasTrans,     CBlasNonUnit, 8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLUT<CBlasConjTrans, CBlasUnit,    8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLUT<CBlasConjTrans, CBlasNonUnit, 8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLN<CBlasUnit,    64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLN<CBlasNonUnit, 64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLT<CBlasTrans,     CBlasUnit,    8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLT<CBlasTrans,     CBlasNonUnit, 8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLT<CBlasConjTrans, CBlasUnit,    8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmLLT<CBlasConjTrans, CBlasNonUnit, 8, 8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);

template void ztrmmRUN<CBlasUnit,    64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRUN<CBlasNonUnit, 64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRUT<CBlasTrans,     CBlasUnit,     8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRUT<CBlasTrans,     CBlasNonUnit,  8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRUT<CBlasConjTrans, CBlasUnit,     8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRUT<CBlasConjTrans, CBlasNonUnit,  8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLN<CBlasUnit,    64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLN<CBlasNonUnit, 64,  4, 16, 16,  4>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLT<CBlasTrans,     CBlasUnit,     8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLT<CBlasTrans,     CBlasNonUnit,  8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLT<CBlasConjTrans, CBlasUnit,     8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
template void ztrmmRLT<CBlasConjTrans, CBlasNonUnit,  8,  8,  4,  4,  8>(const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, cuDoubleComplex, int, int, int, int, int);
