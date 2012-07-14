#include "blas.h"
#include <cuComplex.h>

// y(1:8) += alpha * x(1:8)
__device__ void zaxpy(cuDoubleComplex a, cuDoubleComplex * b, cuDoubleComplex * c) {
  c[0] = cuCfma(a, b[0], c[0]); c[1] = cuCfma(a, b[1], c[1]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex a, cuDoubleComplex * b, cuDoubleComplex * c) {
  c[0] = cuCfma(a, b[0], c[0]); if (1 >= n) return;
  c[1] = cuCfma(a, b[1], c[1]);
}

/**
 * ZTRSM:
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
 * @param bx     blockDim.x.
 * @param by     blockDim.y.
 */
template <CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int bx, unsigned int by>
__global__ void ztrsm(int m, int n,
                      cuDoubleComplex alpha, const cuDoubleComplex * __restrict__ A, int lda,
                      cuDoubleComplex * __restrict__ B, int ldb) {

 if (side == CBlasLeft) {
//     typedef char _x[(mb == 2) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    __shared__ cuDoubleComplex a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ cuDoubleComplex b[mb][nb + 1];
    cuDoubleComplex x[2];

    const int ti = threadIdx.y * bx + threadIdx.x;

    A += threadIdx.y * lda + threadIdx.x;
    B += (blockIdx.y * nb + threadIdx.y) * ldb + threadIdx.x;
    n -= blockIdx.y * nb + threadIdx.y;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) || (uplo == CBlasLower && transA != CBlasNoTrans)) {
      const int mm = m & (mb - 1);
      int i = m - mm;

      A += i * lda + i;
      cuDoubleComplex * X = B + i;

      if (mm > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConj(A[0]);
        }

        __syncthreads();

        switch (mm - 1) {
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        }

        __syncthreads();

        b[0][ti] = x[0]; b[1][ti] = x[1];

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y]; }}
        }
      }

      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuDoubleComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.x][threadIdx.y] = _A[0];
            else
              a[threadIdx.x][threadIdx.y] = cuConj(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConj(A[0]);
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);

        b[0][ti] = x[0]; b[1][ti] = x[1];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];

        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      cuDoubleComplex * X = B;
      int i = 0;

      while (m > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        __syncthreads();

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.x][threadIdx.y] = _A[0];
            else
              a[threadIdx.x][threadIdx.y] = cuConj(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConj(_A[0]);
        }

        __syncthreads();

        if (m < mb) break;

        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]); zaxpy(1, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]);

        b[0][ti] = x[0]; b[1][ti] = x[1];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];

        __syncthreads();

        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
      if (m > 1) { zaxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

      __syncthreads();

      b[0][ti] = x[0]; b[1][ti] = x[1];

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = b[threadIdx.x][by * 0 + threadIdx.y]; if (by * 1 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 1 + threadIdx.y];
      }
    }
  }
  else {
//     typedef char _x[(nb == 2) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 2 elements at a time
    __shared__ cuDoubleComplex a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuDoubleComplex x[2];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in a and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) || (uplo == CBlasLower && transA != CBlasNoTrans)) {
      cuDoubleComplex * X = B;
      int j = 0;

      while (n > 0) {
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = j;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.y][threadIdx.x] = _A[0];
            else
              a[threadIdx.y][threadIdx.x] = cuConj(_A[0]);
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = _A[0];
        else
          a[threadIdx.y][threadIdx.x] = _A[0];

        __syncthreads();

        if (n < nb) break;

        if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        if (n > 1) { zaxpy(7, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      if (n > 0) { if (diag == CBlasNonUnit)x[0] = cuCdiv(x[0], a[0][0]);
      if (n > 1) { zaxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuDoubleComplex * X = B + j * ldb;

      if (nn > 0) {
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.y][threadIdx.x] = A[0];
          else
            a[threadIdx.y][threadIdx.x] = cuConj(A[0]);
        }

        __syncthreads();

        switch (nn - 1) {
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        }

        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn) {
          X[1 * ldb] = x[1]; }
        }
      }

      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuDoubleComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.y][threadIdx.x] = _A[0];
            else
              a[threadIdx.y][threadIdx.x] = cuConj(_A[0]);
          }

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.y][threadIdx.x] = A[0];
          else
            a[threadIdx.y][threadIdx.x] = cuConj(A[0]);
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
        }

        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}

template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit,  2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,     2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit,  2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,     2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit,  2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,     2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit,  2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,     2,  4, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit,  4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,     4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit,  4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,     4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit,  4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,     4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit,  4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,     4,  2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
