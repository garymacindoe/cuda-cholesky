// nvcc -I../../include -O2 -arch=compute_13 -code=sm_13 -use_fast_math -Xptxas=-v -maxrregcount=32 -cubin strsm.cu
#include "blas.h"

__device__ void sscal(float alpha, const float * x, int incx, float * y) {
//   y = alpha * x;
  y[ 0] = alpha * x[ 0 * incx]; y[ 1] = alpha * x[ 1 * incx];
  y[ 2] = alpha * x[ 2 * incx]; y[ 3] = alpha * x[ 3 * incx];
  y[ 4] = alpha * x[ 4 * incx]; y[ 5] = alpha * x[ 5 * incx];
  y[ 6] = alpha * x[ 6 * incx]; y[ 7] = alpha * x[ 7 * incx];
  y[ 8] = alpha * x[ 8 * incx]; y[ 9] = alpha * x[ 9 * incx];
  y[10] = alpha * x[10 * incx]; y[11] = alpha * x[11 * incx];
  y[12] = alpha * x[12 * incx]; y[13] = alpha * x[13 * incx];
  y[14] = alpha * x[14 * incx]; y[15] = alpha * x[15 * incx];
}

__device__ void saxpy(float alpha, const float * x, int incx, float * y) {
//   y -= alpha * x;
  y[ 0] -= alpha * x[ 0 * incx];
  y[ 1] -= alpha * x[ 1 * incx];
  y[ 2] -= alpha * x[ 2 * incx];
  y[ 3] -= alpha * x[ 3 * incx];
  y[ 4] -= alpha * x[ 4 * incx];
  y[ 5] -= alpha * x[ 5 * incx];
  y[ 6] -= alpha * x[ 6 * incx];
  y[ 7] -= alpha * x[ 7 * incx];
  y[ 8] -= alpha * x[ 8 * incx];
  y[ 9] -= alpha * x[ 9 * incx];
  y[10] -= alpha * x[10 * incx];
  y[11] -= alpha * x[11 * incx];
  y[12] -= alpha * x[12 * incx];
  y[13] -= alpha * x[13 * incx];
  y[14] -= alpha * x[14 * incx];
  y[15] -= alpha * x[15 * incx];
}

template <CBlasDiag diag>
__device__ void sscal(int n, float alpha, const float * x, float * y, int incy) {
//   y = (diag == CBlasNonUnit) ? x / alpha : x;
  y[0] = (diag == CBlasNonUnit) ? x[ 0] / alpha : x[ 0]; if ( 1 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 1] / alpha : x[ 1]; if ( 2 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 2] / alpha : x[ 2]; if ( 3 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 3] / alpha : x[ 3]; if ( 4 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 4] / alpha : x[ 4]; if ( 5 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 5] / alpha : x[ 5]; if ( 6 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 6] / alpha : x[ 6]; if ( 7 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 7] / alpha : x[ 7]; if ( 8 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 8] / alpha : x[ 8]; if ( 9 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[ 9] / alpha : x[ 9]; if (10 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[10] / alpha : x[10]; if (11 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[11] / alpha : x[11]; if (12 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[12] / alpha : x[12]; if (13 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[13] / alpha : x[13]; if (14 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[14] / alpha : x[14]; if (15 >= n) return; y += incy;
  y[0] = (diag == CBlasNonUnit) ? x[15] / alpha : x[15];
}


/**
 * STRSM:
 *   B := alpha * inv( A ) * B for side == CBlasLeft, transA == CBlasNoTrans
 *   B := alpha * inv( A') * B for side == CBlasLeft, transA == CBlasTrans
 *   B := alpha * B * inv( A ) for side == CBlasRight, transA == CBlasNoTrans
 *   B := alpha * B * inv( A') for side == CBlasRight, transA == CBlasTrans
 *
 * Only the upper or lower triangle of A is used.
 *
 * @param side   whether A multiplies B from the left or right.
 * @param uplo   uplo for A.
 * @param transA transpose for A.
 * @param diag   whether A is unit or nonunit diagonal.
 * @param mb     the number of rows in the block of B.
 * @param nb     the number of columns in the block of B.
 * @param kb     how far to unroll the inner loop.
 * @param bx     blockDim.x.
 * @param by     blockDim.y.
 */
template <CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int kb,
          unsigned int bx, unsigned int by>
__global__ void strsm(int m, int n,
                      float alpha, const float * __restrict__ A, int lda,
                      float * __restrict__ B, int ldb) {

  float temp[16];

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (int j = 0; j < n; j += 16) {
          for (int i = m - 1; i >= 0; i--) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = i + 1; k < m; k++)
              saxpy(A[k * lda + i], &B[j * ldb + k], ldb, temp);
            sscal<diag>(n, A[i * lda + i], temp, &B[j * ldb + i], ldb);
          }
        }
      }
      else {
        for (int j = 0; j < n; j += 16) {
          for (int i = 0; i < m; i++) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = 0; k < i; k++)
              saxpy(A[k * lda + i], &B[j * ldb + k], ldb, temp);
            sscal<diag>(n, A[i * lda + i], temp, &B[j * ldb + i], ldb);
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (int j = 0; j < n; j += 16) {
          for (int i = 0; i < m; i++) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = 0; k < i; k++)
              saxpy(A[i * lda + k], &B[j * ldb + k], ldb, temp);
            sscal<diag>(n, A[i * lda + i], temp, &B[j * ldb + i], ldb);
          }
        }
      }
      else {
        for (int j = 0; j < n; j += 16) {
          for (int i = m - 1; i >= 0; i--) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = i + 1; k < m; k++)
              saxpy(A[i * lda + k], &B[j * ldb + k], ldb, temp);
            sscal<diag>(n, A[i * lda + i], temp, &B[j * ldb + i], ldb);
          }
        }
      }
    }
  }
  else {
    for (int i = 0; i < m; i++) {

      if (transA == CBlasNoTrans) {
        if (uplo == CBlasUpper) {
          for (int j = 0; j < n; j += 16) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = 0; k < j + threadIdx.x; k++)
              saxpy(temp[k], &A[j * lda + k], lda, temp);
            sscal<diag>(n, A[j * lda + j], temp, &B[j * ldb + i], ldb);
          }
        }
        else {
          for (int j = n - 1; j >= 0; j -= 16) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = j + 1; k < n; k++)
              saxpy(B[k * ldb + i], &A[j * lda + k], lda, temp);
            sscal<diag>(n, A[j * lda + j], temp, &B[j * ldb + i], ldb);
          }
        }
      }
      else {
        if (uplo == CBlasUpper) {
          for (int j = n - 1; j >= 0; j -= 16) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = j + 1; k < n; k++)
              saxpy(B[k * ldb + i], &A[k * lda + j], 1, temp);
            sscal<diag>(n, A[j * lda + j], temp, &B[j * ldb + i], ldb);
          }
        }
        else {
          for (int j = 0; j < n; j += 16) {
            sscal(alpha, &B[j * ldb + i], ldb, temp);
            for (int k = 0; k < j; k++)
              saxpy(B[k * ldb + i], &A[k * lda + j], 1, temp);
            sscal<diag>(n, A[j * lda + j], temp, &B[j * ldb + i], ldb);
          }
        }
      }

    }
  }
}

template void strsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    64, 16, 16, 16,  4>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    32, 32,  8,  8,  8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
