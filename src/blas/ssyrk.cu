#include "blas.h"

// y(1:16) += alpha * x(1:16)
__device__ void saxpy(float alpha, const float * x, float * y) {
  y[ 0] += alpha * x[ 0]; y[ 1] += alpha * x[ 1]; y[ 2] += alpha * x[ 2]; y[ 3] += alpha * x[ 3];
  y[ 4] += alpha * x[ 4]; y[ 5] += alpha * x[ 5]; y[ 6] += alpha * x[ 6]; y[ 7] += alpha * x[ 7];
  y[ 8] += alpha * x[ 8]; y[ 9] += alpha * x[ 9]; y[10] += alpha * x[10]; y[11] += alpha * x[11];
  y[12] += alpha * x[12]; y[13] += alpha * x[13]; y[14] += alpha * x[14]; y[15] += alpha * x[15];
}

/**
 * SSYRK:
 *   C := alpha * AA' + beta * C for trans == CBlasNoTrans
 *   C := alpha * A'A + beta * C for trans != CBlasNoTrans
 *
 * Only the upper or lower triangle of C is updated.
 *
 * @param uplo   uplo for C.
 * @param trans  transpose for A.
 * @param mb     the number of rows in the block of C.
 * @param nb     the number of columns in the block of C.
 * @param kb     how far to unroll the inner loop.
 * @param bx     blockDim.x.
 * @param by     blockDim.y.
 */
template <CBlasUplo uplo, CBlasTranspose trans,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void ssyrk(int n, int k,
                      float alpha, const float * __restrict__ A, int lda,
                      float beta, float * __restrict__ C, int ldc) {

//   int bi, bj, nnb = (n + nb - 1) / nb;
//   if (uplo == CBlasLower) {
//     bi = blockIdx.x % nnb;
//     bj = blockIdx.x / nnb;
//     if (bi < bj) {
//       bi = nnb - bi - 1;
//       bj = nnb - bj;
//     }
//   }
//   else {
//     bi = blockIdx.x / nnb;
//     bj = blockIdx.x % nnb;
//     if (bj < bi) {
//       bi = nnb - bi;
//       bj = nnb - bj - 1;
//     }
//   }
//
//   bi *= mb;
//   bj *= nb;

  // Using an SGEMM kernel, SSYRK is:
  // C = alpha * A * B + beta * C
  // with A = A and B = A' when trans == CBlasNoTrans, and
  // with A = A' and B = A when trans == CBlasTrans
  const float * __restrict__ B = A;

  const int bi = blockIdx.x * mb;       // Starting row of block of C
  const int bj = blockIdx.y * nb;       // Starting column of block of C
  const int ti = threadIdx.y * bx + threadIdx.x;        // Unwrapped thread index [0, bx * by]

  if (uplo == CBlasUpper) {
    if (bj + nb - 1 < bi)
      return;
  }
  else if (uplo == CBlasLower) {
    if (bi + mb - 1 < bj)
      return;
  }

  /*
   * Compute our starting points in A, B and C.
   *
   * For trans == CBlasTrans A is cached in shared memory so the unwrapped
   * thread index can be re-wrapped around mb when calculating C.
   *
   * If trans == CBlasNoTrans then bx * by == mb (checked later on) so there
   * doesn't need to be a separate check for trans == CBlasNoTrans in
   * calculating the start of C here.
   */
  if (trans == CBlasNoTrans) {
    A += bi + ti;
    B += threadIdx.y * lda + bj + threadIdx.x;
    C += bj * ldc + bi + ti;
  }
  else {
    A += (bi + threadIdx.y) * lda + threadIdx.x;
    B += (bj + threadIdx.y) * lda + threadIdx.x;
    C += (bj + 16 * (ti / mb)) * ldc + bi + ti % mb;
  }

  /*
   * Blocks of A and B in shared memory and C in registers.
   */
  __shared__ float a[mb][kb + 1];       // Optimised away when trans == CBlasNoTrans
  __shared__ float b[kb][(trans == CBlasNoTrans) ? nb : nb + 1];

  float c[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  while (k > 0) {
    if (trans == CBlasNoTrans) {
      // C = aAA' + bC so read B into shared memory and transpose leaving A
      // untransposed in global memory
#pragma unroll
      for (int l = 0; l < kb; l += by) {
#pragma unroll
        for (int j = 0; j < nb; j += bx)
          b[l + threadIdx.y][j + threadIdx.x] = B[l * lda + j];
      }
    }
    else {
      // C = aA'A + bC so read A into shared memory and transpose before reading
      // B into shared memory untransposed
#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int i = 0; i < mb; i += by)
          a[i + threadIdx.y][l + threadIdx.x] = A[i * lda + l];
      }
      A += kb;

#pragma unroll
      for (int l = 0; l < kb; l += bx) {
#pragma unroll
        for (int j = 0; j < nb; j += by)
          b[l + threadIdx.x][j + threadIdx.y] = B[j * lda + l];
      }
    }

    __syncthreads();

    if (k < kb) break;

    if (trans == CBlasNoTrans) {
      // Read A from global memory
#pragma unroll
      for (int l = 0; l < kb; l++) {
        saxpy(A[0], b[l], c);
        A += lda;
      }
    }
    else {
      // Read A from shared memory
#pragma unroll
      for (int l = 0; l < kb; l++)
        saxpy(a[ti % mb][l], &b[l][16 * (ti / mb)], c);
    }

    __syncthreads();

    B += (trans != CBlasNoTrans) ? kb : kb * lda;
    k -= kb;
  }

  if (trans == CBlasNoTrans) {
    // Read A from global memory
    for (int l = 0; l < k; l++) {
      saxpy(A[0], b[l], c);
      A += lda;
    }
  }
  else {
    // Read A' from shared memory
    for (int l = 0; l < k; l++)
      saxpy(a[ti % mb][l], &b[l][16 * (ti / mb)], c);
  }

  if (bx * by == mb)
    n -= bj;
  else
    n -= bj + 16 * (ti / mb);
  if (n <= 0) return;
  if ((bx * by == mb && bi + ti < n) || (bx * by > mb && bi + ti % mb < n)) {
    if (beta == 0.0f) {
      C[0] = alpha * c[ 0]; if ( 1 >= n) return; C += ldc;
      C[0] = alpha * c[ 1]; if ( 2 >= n) return; C += ldc;
      C[0] = alpha * c[ 2]; if ( 3 >= n) return; C += ldc;
      C[0] = alpha * c[ 3]; if ( 4 >= n) return; C += ldc;
      C[0] = alpha * c[ 4]; if ( 5 >= n) return; C += ldc;
      C[0] = alpha * c[ 5]; if ( 6 >= n) return; C += ldc;
      C[0] = alpha * c[ 6]; if ( 7 >= n) return; C += ldc;
      C[0] = alpha * c[ 7]; if ( 8 >= n) return; C += ldc;
      C[0] = alpha * c[ 8]; if ( 9 >= n) return; C += ldc;
      C[0] = alpha * c[ 9]; if (10 >= n) return; C += ldc;
      C[0] = alpha * c[10]; if (11 >= n) return; C += ldc;
      C[0] = alpha * c[11]; if (12 >= n) return; C += ldc;
      C[0] = alpha * c[12]; if (13 >= n) return; C += ldc;
      C[0] = alpha * c[13]; if (14 >= n) return; C += ldc;
      C[0] = alpha * c[14]; if (15 >= n) return; C += ldc;
      C[0] = alpha * c[15];
    }
    else {
      C[0] = alpha * c[ 0] + beta * C[0]; if ( 1 >= n) return; C += ldc;
      C[0] = alpha * c[ 1] + beta * C[0]; if ( 2 >= n) return; C += ldc;
      C[0] = alpha * c[ 2] + beta * C[0]; if ( 3 >= n) return; C += ldc;
      C[0] = alpha * c[ 3] + beta * C[0]; if ( 4 >= n) return; C += ldc;
      C[0] = alpha * c[ 4] + beta * C[0]; if ( 5 >= n) return; C += ldc;
      C[0] = alpha * c[ 5] + beta * C[0]; if ( 6 >= n) return; C += ldc;
      C[0] = alpha * c[ 6] + beta * C[0]; if ( 7 >= n) return; C += ldc;
      C[0] = alpha * c[ 7] + beta * C[0]; if ( 8 >= n) return; C += ldc;
      C[0] = alpha * c[ 8] + beta * C[0]; if ( 9 >= n) return; C += ldc;
      C[0] = alpha * c[ 9] + beta * C[0]; if (10 >= n) return; C += ldc;
      C[0] = alpha * c[10] + beta * C[0]; if (11 >= n) return; C += ldc;
      C[0] = alpha * c[11] + beta * C[0]; if (12 >= n) return; C += ldc;
      C[0] = alpha * c[12] + beta * C[0]; if (13 >= n) return; C += ldc;
      C[0] = alpha * c[13] + beta * C[0]; if (14 >= n) return; C += ldc;
      C[0] = alpha * c[14] + beta * C[0]; if (15 >= n) return; C += ldc;
      C[0] = alpha * c[15] + beta * C[0];
    }
  }
}

/**
 * For C = aAB + bC:
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
template void ssyrk<CBlasUpper, CBlasNoTrans, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float, float * __restrict__, int);
template void ssyrk<CBlasLower, CBlasNoTrans, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float, float * __restrict__, int);
template void ssyrk<CBlasUpper, CBlasTrans,   32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float, float * __restrict__, int);
template void ssyrk<CBlasLower, CBlasTrans,   32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float, float * __restrict__, int);
