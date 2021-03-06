#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && (!defined(__BANK_CONFLICTS__) || __BANK_CONFLICTS__ <= 1)

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

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * CGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
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
__global__ void cgemm2(const cuComplex * __restrict__ A, const cuComplex * __restrict__ B,
                       const cuComplex * __restrict__ C, cuComplex * __restrict__ D,
                       cuComplex alpha, cuComplex beta,
                       int lda, int ldb, int ldc, int ldd,
                       int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (transA != CBlasNoTrans) {
    tj = 8 * (ti / mb);
    ti = ti % mb;
  }

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
  A += (transA == CBlasNoTrans) ? bi + ti : (bi + threadIdx.y) * lda + threadIdx.x;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  D += (bj + tj) * ldd + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ float a_real[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ float a_imag[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ float b_real[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];
  __shared__ float b_imag[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuComplex d[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  while (k > 0) {
    // If A is to be transposed cache it in shared memory
    if (transA != CBlasNoTrans) {
      if (transA == CBlasConjTrans) {
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

    // B will always be "transposed" w.r.t. C so must always be cached in shared
    // memory (i.e. it is read along the K or N dimensions when M is the
    // dimension being expanded).
    if (transB == CBlasNoTrans) {
#pragma unroll
      for (int j = 0; j < nb; j += by) {
        b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(B[j * ldb]);
        b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(B[j * ldb]);
      }
    }
    else if (transB == CBlasConjTrans) {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] =  cuCrealf(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = -cuCimagf(B[l * ldb + j]);
        }
      }
    }
    else {
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx) {
          b_real[l + threadIdx.y][j + threadIdx.x] = cuCrealf(B[l * ldb + j]);
          b_imag[l + threadIdx.y][j + threadIdx.x] = cuCimagf(B[l * ldb + j]);
        }
      }
    }

    __syncthreads();

    if (k < kb) break;

    if (transA == CBlasNoTrans) {
      // Read A straight from global memory.
#pragma unroll
      for (int l = 0; l < kb; l++) {
        caxpy(A[0], b_real[l], b_imag[l], d);
        A += lda;
      }
    }
    else {
      // Read A from shared memory.
#pragma unroll
      for (int l = 0; l < kb; l++)
        caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]),
              &b_real[l][tj], &b_imag[l][tj], d);
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  if (transA == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      caxpy(A[0], b_real[l], b_imag[l], d);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      caxpy(make_cuComplex(a_real[ti][l], a_imag[ti][l]),
            &b_real[l][tj], &b_imag[l][tj], d);
  }

  if (n <= 0 || m <= 0) return;
  if (cuCrealf(beta) == 0.0f && cuCimagf(beta) == 0.0f) {
    D[0] = cuCmulf(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[1]); if (2 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[2]); if (3 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[3]); if (4 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[4]); if (5 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[5]); if (6 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[6]); if (7 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[7]);
  }
  else {
    D[0] = cuCfmaf(alpha, d[0], cuCmulf(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[1], cuCmulf(beta, C[0])); if (2 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[2], cuCmulf(beta, C[0])); if (3 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[3], cuCmulf(beta, C[0])); if (4 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[4], cuCmulf(beta, C[0])); if (5 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[5], cuCmulf(beta, C[0])); if (6 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[6], cuCmulf(beta, C[0])); if (7 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[7], cuCmulf(beta, C[0]));
  }
}

#else

// y(1:8) += alpha * x(1:8)
__device__ void caxpy(cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCfmaf(alpha, x[0], y[0]); y[1] = cuCfmaf(alpha, x[1], y[1]);
  y[2] = cuCfmaf(alpha, x[2], y[2]); y[3] = cuCfmaf(alpha, x[3], y[3]);
  y[4] = cuCfmaf(alpha, x[4], y[4]); y[5] = cuCfmaf(alpha, x[5], y[5]);
  y[6] = cuCfmaf(alpha, x[6], y[6]); y[7] = cuCfmaf(alpha, x[7], y[7]);
}

/**
 * This implementation is out-of-place.  For in-place call with D = C and ldd = ldc.
 *
 * CGEMM:
 *   D := alpha * AB   + beta * C for transA == CBlasNoTrans and transB == CBlasNoTrans
 *   D := alpha * AB'  + beta * C for transA == CBlasNoTrans and transB == CBlasTrans
 *   D := alpha * AB^  + beta * C for transA == CBlasNoTrans and transB == CBlasConjTrans
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
__global__ void cgemm2(const cuComplex * __restrict__ A, const cuComplex * __restrict__ B,
                       const cuComplex * __restrict__ C, cuComplex * __restrict__ D,
                       cuComplex alpha, cuComplex beta,
                       int lda, int ldb, int ldc, int ldd,
                       int m, int n, int k) {

  const int bi = blockIdx.x * mb;       // Starting row of block of C/D
  const int bj = blockIdx.y * nb;       // Starting column of block of C/D
  int ti = threadIdx.y * bx + threadIdx.x;
  int tj = 0;
  if (transA != CBlasNoTrans) {
    tj = 8 * (ti / mb);
    ti = ti % mb;
  }

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
  A += (transA == CBlasNoTrans) ? bi + ti : (bi + threadIdx.y) * lda + threadIdx.x;
  B += (transB == CBlasNoTrans) ? (bj + threadIdx.y) * ldb + threadIdx.x : threadIdx.y * ldb + bj + threadIdx.x;
  C += (bj + tj) * ldc + bi + ti;
  D += (bj + tj) * ldd + bi + ti;
  n -= bj + tj;
  m -= bi + ti;

  /*
   * Blocks of A and B in shared memory and D in registers.
   */
  __shared__ cuComplex a[mb][kb + 1];       // Optimised away when transA == CBlasNoTrans
  __shared__ cuComplex b[kb][(transB == CBlasNoTrans) ? nb + 1 : nb];

  cuComplex d[] = { { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f },
                    { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 0.0f } };

  while (k > 0) {
    // If A is to be transposed cache it in shared memory
    if (transA != CBlasNoTrans) {
      if (transA == CBlasConjTrans) {
#pragma unroll
        for (int l = 0; l < kb; l += bx) {
#pragma unroll
          for (int i = 0; i < mb; i += by)
            a[i + threadIdx.y][l + threadIdx.x] = cuConjf(A[i * lda + l]);
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
    }

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
          b[l + threadIdx.y][j + threadIdx.x] = cuConjf(B[l * ldb + j]);
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

    if (transA == CBlasNoTrans) {
      // Read A straight from global memory.
#pragma unroll
      for (int l = 0; l < kb; l++) {
        caxpy(A[0], b[l], d);
        A += lda;
      }
    }
    else {
      // Read A from shared memory.
#pragma unroll
      for (int l = 0; l < kb; l++)
        caxpy(a[ti][l], &b[l][tj], d);
    }

    __syncthreads();

    B += (transB == CBlasNoTrans) ? kb : kb * ldb;
    k -= kb;
  }

  if (transA == CBlasNoTrans) {
    for (int l = 0; l < k; l++) {
      caxpy(A[0], b[l], d);
      A += lda;
    }
  }
  else {
    for (int l = 0; l < k; l++)
      caxpy(a[ti][l], &b[l][tj], d);
  }

  if (n <= 0 || m <= 0) return;
  if (cuCrealf(beta) == 0.0f && cuCimagf(beta) == 0.0f) {
    D[0] = cuCmulf(alpha, d[0]); if (1 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[1]); if (2 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[2]); if (3 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[3]); if (4 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[4]); if (5 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[5]); if (6 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[6]); if (7 >= n) return; D += ldd;
    D[0] = cuCmulf(alpha, d[7]);
  }
  else {
    D[0] = cuCfmaf(alpha, d[0], cuCmulf(beta, C[0])); if (1 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[1], cuCmulf(beta, C[0])); if (2 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[2], cuCmulf(beta, C[0])); if (3 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[3], cuCmulf(beta, C[0])); if (4 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[4], cuCmulf(beta, C[0])); if (5 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[5], cuCmulf(beta, C[0])); if (6 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[6], cuCmulf(beta, C[0])); if (7 >= n) return; C += ldc; D += ldd;
    D[0] = cuCfmaf(alpha, d[7], cuCmulf(beta, C[0]));
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
 *        (nb + 1)*kb*sizeof(float) is less than the amount of shared memory
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
template __global__ void cgemm2<CBlasNoTrans,   CBlasNoTrans,   64,  8, 16, 16,  4>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasNoTrans,   CBlasTrans,     64,  8, 16,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasNoTrans,   CBlasConjTrans, 64,  8, 16,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasTrans,     CBlasNoTrans,   32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasTrans,     CBlasTrans,     32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasTrans,     CBlasConjTrans, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasConjTrans, CBlasNoTrans,   32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasConjTrans, CBlasTrans,     32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
template __global__ void cgemm2<CBlasConjTrans, CBlasConjTrans, 32, 16,  8,  8,  8>(const cuComplex * __restrict__, const cuComplex * __restrict__, const cuComplex * __restrict__, cuComplex * __restrict__, cuComplex, cuComplex, int, int, int, int, int, int, int);
