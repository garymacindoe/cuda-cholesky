#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy4(cuDoubleComplex alpha, const int * __restrict__ x_real_hi, const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi, const int * __restrict__ x_imag_lo, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[0], x_real_lo[0]),
                     __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);

  y[1] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[1], x_real_lo[1]),
                     __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);

  y[2] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[2], x_real_lo[2]),
                     __hiloint2double(x_imag_hi[2], x_imag_lo[2])), y[2]);

  y[3] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[3], x_real_lo[3]),
                     __hiloint2double(x_imag_hi[3], x_imag_lo[3])), y[3]);
}

// y(1:2) += alpha * x(1:2)
__device__ void zaxpy2(cuDoubleComplex alpha, const int * __restrict__ x_real_hi, const int * __restrict__ x_real_lo,
                       const int * __restrict__ x_imag_hi, const int * __restrict__ x_imag_lo, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[0], x_real_lo[0]),
                     __hiloint2double(x_imag_hi[0], x_imag_lo[0])), y[0]);

  y[1] = cuCfma(alpha, make_cuDoubleComplex(
                     __hiloint2double(x_real_hi[1], x_real_lo[1]),
                     __hiloint2double(x_imag_hi[1], x_imag_lo[1])), y[1]);
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
 *
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2N(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += bi + ti;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj) * ldc + bi + ti;
  D += (bj) * ldd + bi + ti;
  n -= bj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ int b_real_hi[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_real_lo[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_imag_hi[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_imag_lo[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
//       typedef char x[bx == kb ? 1 : -1];
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
        b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
        b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
        b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint( cuCreal(B[l * ldb + j]));
          b_real_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint( cuCreal(B[l * ldb + j]));
          b_imag_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(-cuCimag(B[l * ldb + j]));
          b_imag_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(-cuCimag(B[l * ldb + j]));
        }
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(cuCreal(B[l * ldb + j]));
          b_real_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(cuCreal(B[l * ldb + j]));
          b_imag_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(cuCimag(B[l * ldb + j]));
          b_imag_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(cuCimag(B[l * ldb + j]));
        }
      }
    }

    __syncthreads();

    if (k < kb) break;

      // Read A straight from global memory.
#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy4(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], d);
      A += lda;
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy4(A[0], b_real_hi[l], b_real_lo[l], b_imag_hi[l], b_imag_lo[l], d);
    A += lda;
  }

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]); if (2 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[2]); if (3 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[3]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0])); if (2 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[2], cuCmul(beta, C[0])); if (3 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[3], cuCmul(beta, C[0]));
  }
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * A'B  + beta * C for transA == CBlasTrans and transB == CBlasNoTrans
 *   D := alpha * A'B' + beta * C for transA == CBlasTrans and transB == CBlasTrans
 *   D := alpha * A'B^ + beta * C for transA == CBlasTrans and transB == CBlasConjTrans
 *   D := alpha * A^B  + beta * C for transA == CBlasConjTrans and transB == CBlasNoTrans
 *   D := alpha * A^B' + beta * C for transA == CBlasConjTrans and transB == CBlasTrans
 *   D := alpha * A^B^ + beta * C for transA == CBlasConjTrans and transB == CBlasConjTrans
 *
 * @param transA  transpose for A.
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transA, CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2T(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 2 * (ti / mb);
  ti = ti % mb;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  D += (bj + tj) * ldd + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ int a_real_hi[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ int a_real_lo[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ int a_imag_hi[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ int a_imag_lo[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ int b_real_hi[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_real_lo[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_imag_hi[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ int b_imag_lo[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // If A is to be transposed cache it in shared memory
    if (transA == CBlasConjTrans) {
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

    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
//       typedef char x[bx == kb ? 1 : -1];
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(B[j * ldb]));
        b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(B[j * ldb]));
        b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(B[j * ldb]));
        b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(B[j * ldb]));
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(cuCreal(B[l * ldb + j]));
          b_real_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(cuCreal(B[l * ldb + j]));
          b_imag_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(-cuCimag(B[l * ldb + j]));
          b_imag_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(-cuCimag(B[l * ldb + j]));
        }
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(cuCreal(B[l * ldb + j]));
          b_real_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(cuCreal(B[l * ldb + j]));
          b_imag_hi[l + threadIdx.y][j + threadIdx.x] = __double2hiint(cuCimag(B[l * ldb + j]));
          b_imag_lo[l + threadIdx.y][j + threadIdx.x] = __double2loint(cuCimag(B[l * ldb + j]));
        }
      }
    }

    __syncthreads();

    if (k < kb) break;

    // Read A from shared memory.
#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy2(make_cuDoubleComplex(
                            __hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                            __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
            &b_real_hi[l][tj], &b_real_lo[l][tj],
            &b_imag_hi[l][tj], &b_imag_lo[l][tj], d);

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    zaxpy2(make_cuDoubleComplex(
                          __hiloint2double(a_real_hi[ti][l], a_real_lo[ti][l]),
                          __hiloint2double(a_imag_hi[ti][l], a_imag_lo[ti][l])),
            &b_real_hi[l][tj], &b_real_lo[l][tj],
            &b_imag_hi[l][tj], &b_imag_lo[l][tj], d);

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0]));
  }
}

#elif __CUDA_ARCH__ >= 200 && __CUDA_ARCH__ < 210 || __CUDA_ARCH__ < 200 && __BANK_CONFLICTS__ == 2

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy4(cuDoubleComplex alpha, const double * x_real, const double * x_imag, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(x_real[0], x_imag[0]), y[0]);
  y[1] = cuCfma(alpha, make_cuDoubleComplex(x_real[1], x_imag[1]), y[1]);
  y[2] = cuCfma(alpha, make_cuDoubleComplex(x_real[2], x_imag[2]), y[2]);
  y[3] = cuCfma(alpha, make_cuDoubleComplex(x_real[3], x_imag[3]), y[3]);
}

// y(1:2) += alpha * x(1:2)
__device__ void zaxpy2(cuDoubleComplex alpha, const double * x_real, const double * x_imag, cuDoubleComplex * y) {
  y[0] = cuCfma(alpha, make_cuDoubleComplex(x_real[0], x_imag[0]), y[0]);
  y[1] = cuCfma(alpha, make_cuDoubleComplex(x_real[1], x_imag[1]), y[1]);
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
 *
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2N(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  const int ti = threadIdx.y * bx + threadIdx.x;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += bi + ti;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += bj * ldc + bi + ti;
  D += bj * ldd + bi + ti;
  n -= bj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ double b_real[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ double b_imag[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real[threadIdx.x][j + threadIdx.y] = cuCreal(B[j * ldb]);
        b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(B[j * ldb]);
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] =  cuCreal(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = -cuCimag(B[l * ldb + j]);
        }
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] = cuCreal(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = cuCimag(B[l * ldb + j]);
        }
      }
    }

    __syncthreads();

    if (k < kb) break;

    // Read A straight from global memory.
#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy4(A[0], b_real[l], b_imag[l], d);
      A += lda;
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy4(A[0], b_real[l], b_imag[l], d);
    A += lda;
  }

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]); if (2 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[2]); if (3 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[3]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0])); if (2 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[2], cuCmul(beta, C[0])); if (3 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[3], cuCmul(beta, C[0]));
  }
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
 *
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transA, CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2T(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;
  const int tj = 2 * (ti / mb);
  ti = ti % mb;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  D += (bj + tj) * ldd + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ double a_real[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ double a_imag[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ double b_real[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ double b_imag[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // If A is to be transposed cache it in shared memory
    if (transA != CBlasNoTrans) {
      if (transA == CBlasConjTrans) {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real[i + threadIdx.y][threadIdx.x] =  cuCreal(A[i * lda]);
          a_imag[i + threadIdx.y][threadIdx.x] = -cuCimag(A[i * lda]);
        }
      }
      else {
#pragma unroll
        for (int i = 0; i < mb; i += by) {
          a_real[i + threadIdx.y][threadIdx.x] = cuCreal(A[i * lda]);
          a_imag[i + threadIdx.y][threadIdx.x] = cuCimag(A[i * lda]);
        }
      }
      A += kb;
    }

    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real[threadIdx.x][j + threadIdx.y] = cuCreal(B[j * ldb]);
        b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(B[j * ldb]);
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] =  cuCreal(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = -cuCimag(B[l * ldb + j]);
        }
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] = cuCreal(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = cuCimag(B[l * ldb + j]);
        }
      }
    }

    __syncthreads();

    if (k < kb) break;

    // Read A from shared memory.
#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy2(make_cuDoubleComplex(a_real[ti][l], a_imag[ti][l]),
            &b_real[l][tj], &b_imag[l][tj], d);

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    zaxpy2(make_cuDoubleComplex(a_real[ti][l], a_imag[ti][l]),
          &b_real[l][tj], &b_imag[l][tj], d);

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0]));
  }
}

#else

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy4(cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ x, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, x[0], y[0]); y[1] = cuCfma(alpha, x[1], y[1]);
  y[2] = cuCfma(alpha, x[2], y[2]); y[3] = cuCfma(alpha, x[3], y[3]);
}

// y(1:2) += alpha * x(1:2)
__device__ void zaxpy2(cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ x, cuDoubleComplex * __restrict__ y) {
  y[0] = cuCfma(alpha, x[0], y[0]); y[1] = cuCfma(alpha, x[1], y[1]);
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
 *
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2N(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += bi + ti;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += bj * ldc + bi + ti;
  D += bj * ldd + bi + ti;
  n -= bj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ cuDoubleComplex b[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          b[l + threadIdx.x][j + threadIdx.y] = B[j * ldb + l];
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = cuConj(B[l * ldb + j]);
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = B[l * ldb + j];
      }
    }

    __syncthreads();

    if (k < kb) break;

    // Read A straight from global memory.
#pragma unroll
    for (int l = 0; l < kb; l++) {
      zaxpy4(A[0], b[l], d);
      A += lda;
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++) {
    zaxpy4(A[0], b[l], d);
    A += lda;
  }

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]); if (2 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[2]); if (3 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[3]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0])); if (2 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[2], cuCmul(beta, C[0])); if (3 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[3], cuCmul(beta, C[0]));
  }
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * ZGEMM:
 *   D := alpha * A'B  + beta * C for transA == CBlasTrans and transB == CBlasNoTrans
 *   D := alpha * A'B' + beta * C for transA == CBlasTrans and transB == CBlasTrans
 *   D := alpha * A'B^ + beta * C for transA == CBlasTrans and transB == CBlasConjTrans
 *   D := alpha * A^B  + beta * C for transA == CBlasConjTrans and transB == CBlasNoTrans
 *   D := alpha * A^B' + beta * C for transA == CBlasConjTrans and transB == CBlasTrans
 *   D := alpha * A^B^ + beta * C for transA == CBlasConjTrans and transB == CBlasConjTrans
 *
 * @param transA  transpose for A.
 * @param transB  transpose for B.
 * @param mb      the number of rows in the block of C/D.
 * @param nb      the number of columns in the block of C/D.
 * @param kb      how far to unroll the inner loop.
 * @param bx      blockDim.x.
 * @param by      blockDim.y.
 */
template <CBlasTranspose transA, CBlasTranspose transB,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void zgemm2T(cuDoubleComplex alpha, cuDoubleComplex beta,
                        const cuDoubleComplex * __restrict__ A, const cuDoubleComplex * __restrict__ B,
                        const cuDoubleComplex * __restrict__ C, cuDoubleComplex * __restrict__ D,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 2 * (ti / mb);
  ti = ti % mb;

  /*
   * Compute our starting points in A, B, C and D.
   *
   * For transA != CBlasNoTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating D.
   *
   * If transA == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for transA == CBlasNoTrans in
   * calculating the start of C/D here.
   */
  A += (bi + threadIdx.y) * lda + threadIdx.x;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  D += (bj + tj) * ldd + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ cuDoubleComplex a[mb][kb + 1];
  __shared__ cuDoubleComplex b[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuDoubleComplex d[] = { { 0.0, 0.0 }, { 0.0, 0.0 } };

  while (k > 0) {
    // If A is to be transposed cache it in shared memory
    if (transA == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][l + threadIdx.x] = cuConj(A[i * lda + l]);
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][l + threadIdx.x] = A[i * lda + l];
      }
    }
    A += kb;

    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          b[l + threadIdx.x][j + threadIdx.y] = B[j * ldb + l];
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = cuConj(B[l * ldb + j]);
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = B[l * ldb + j];
      }
    }

    __syncthreads();

    if (k < kb) break;

      // Read A from shared memory.
#pragma unroll
    for (int l = 0; l < kb; l++)
      zaxpy2(a[ti][l], &b[l][tj], d);

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  for (int l = 0; l < k; l++)
    zaxpy2(a[ti][l], &b[l][tj], d);

  if (n <= 0 || m <= 0) return;
  if (cuCreal(beta) == 0.0 && cuCimag(beta) == 0.0) {
    D[0] = cuCmul(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmul(alpha, d[1]);
  }
  else {
    D[0] = cuCfma(alpha, d[0], cuCmul(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfma(alpha, d[1], cuCmul(beta, C[0]));
  }
}

#endif

/**
 * For D = aAB + bC:
 *   mb must be a multiple of the warp size (32) and less than or equal to the
 *        maximum number of threads per block (512).
 *   nb must be less than or equal to 20 (registers start spilling to global
 *        memory after 20).
 *   kb must be a multiple of the half-warp size (16) and such that
 *        (nb + 1)*kb*sizeof(double) is less than the amount of shared memory
 *        available per block (16384 bytes).
 *
 * mb and nb must be selected such that the bandwidth reduction is greater than
 * the flop:word ratio of the GPU.  The bandwidth reduction for all valid values
 * of mb and nb can be calculated with the following loop (bash):
 * echo -n " mb\nb"; for nb in {1..20}; do printf "%6d" ${nb}; done; echo; for mb in {32..512..32}; do printf "%6d"  ${mb}; for nb in {1..20}; do printf "%6.2f" $(echo 2 / \(1/${mb} + 1/${nb}\) | bc -l); done; echo; done
 *
 * Sample output:
 *  mb\nb     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20
 *     32  1.94  3.76  5.49  7.11  8.65 10.11 11.49 12.80 14.05 15.24 16.37 17.45 18.49 19.48 20.43 21.33 22.20 23.04 23.84 24.62
 *     64  1.97  3.88  5.73  7.53  9.28 10.97 12.62 14.22 15.78 17.30 18.77 20.21 21.61 22.97 24.30 25.60 26.86 28.10 29.30 30.48
 *     96  1.98  3.92  5.82  7.68  9.50 11.29 13.05 14.77 16.46 18.11 19.74 21.33 22.90 24.44 25.95 27.43 28.88 30.32 31.72 33.10
 *    128  1.98  3.94  5.86  7.76  9.62 11.46 13.27 15.06 16.82 18.55 20.26 21.94 23.60 25.24 26.85 28.44 30.01 31.56 33.09 34.59
 *    160  1.99  3.95  5.89  7.80  9.70 11.57 13.41 15.24 17.04 18.82 20.58 22.33 24.05 25.75 27.43 29.09 30.73 32.36 33.97 35.56
 *    192  1.99  3.96  5.91  7.84  9.75 11.64 13.51 15.36 17.19 19.01 20.81 22.59 24.35 26.10 27.83 29.54 31.23 32.91 34.58 36.23
 *    224  1.99  3.96  5.92  7.86  9.78 11.69 13.58 15.45 17.30 19.15 20.97 22.78 24.57 26.35 28.12 29.87 31.60 33.32 35.03 36.72
 *    256  1.99  3.97  5.93  7.88  9.81 11.73 13.63 15.52 17.39 19.25 21.09 22.93 24.74 26.55 28.34 30.12 31.88 33.64 35.37 37.10
 *    288  1.99  3.97  5.94  7.89  9.83 11.76 13.67 15.57 17.45 19.33 21.19 23.04 24.88 26.70 28.51 30.32 32.10 33.88 35.65 37.40
 *    320  1.99  3.98  5.94  7.90  9.85 11.78 13.70 15.61 17.51 19.39 21.27 23.13 24.98 26.83 28.66 30.48 32.28 34.08 35.87 37.65
 *    352  1.99  3.98  5.95  7.91  9.86 11.80 13.73 15.64 17.55 19.45 21.33 23.21 25.07 26.93 28.77 30.61 32.43 34.25 36.05 37.85
 *    384  1.99  3.98  5.95  7.92  9.87 11.82 13.75 15.67 17.59 19.49 21.39 23.27 25.15 27.02 28.87 30.72 32.56 34.39 36.21 38.02
 *    416  2.00  3.98  5.96  7.92  9.88 11.83 13.77 15.70 17.62 19.53 21.43 23.33 25.21 27.09 28.96 30.81 32.67 34.51 36.34 38.17
 *    448  2.00  3.98  5.96  7.93  9.89 11.84 13.78 15.72 17.65 19.56 21.47 23.37 25.27 27.15 29.03 30.90 32.76 34.61 36.45 38.29
 *    480  2.00  3.98  5.96  7.93  9.90 11.85 13.80 15.74 17.67 19.59 21.51 23.41 25.31 27.21 29.09 30.97 32.84 34.70 36.55 38.40
 *    512  2.00  3.98  5.97  7.94  9.90 11.86 13.81 15.75 17.69 19.62 21.54 23.45 25.36 27.25 29.15 31.03 32.91 34.78 36.64 38.50
 *
 * The number of registers per block is mb*32 (compiled with -maxrregcount=32).
 * More threads == better performance (from flop-test) therefore mb is chosen to
 * be the largest number of threads such that the number of blocks per
 * multiprocessor is still limited by the register usage.
 * kb is chosen to be the largest multiple of 16 such that the number of blocks
 * per multiprocessor is limited by the register usage.
 */
template __global__ void zgemm2N<CBlasNoTrans,   64,  4, 16, 16,  4>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2N<CBlasTrans,     64,  4, 16,  4, 16>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2N<CBlasConjTrans, 64,  4, 16,  4, 16>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);

template __global__ void zgemm2T<CBlasTrans,     CBlasNoTrans,    8,  8,  4,  4,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2T<CBlasTrans,     CBlasTrans,      8, 16,  8,  8,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2T<CBlasTrans,     CBlasConjTrans,  8, 16,  8,  8,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2T<CBlasConjTrans, CBlasNoTrans,    8,  8,  4,  4,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2T<CBlasConjTrans, CBlasTrans,      8, 16,  8,  8,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
template __global__ void zgemm2T<CBlasConjTrans, CBlasConjTrans,  8, 16,  8,  8,  8>(cuDoubleComplex, cuDoubleComplex, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, const cuDoubleComplex * __restrict__, cuDoubleComplex * __restrict__, int, int, int, int, int, int, int);
