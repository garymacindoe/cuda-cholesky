#include "blas.h"
#include <cuComplex.h>

// y(1:8) += alpha * x(1:8)
__device__ void caxpy(cuComplex a, cuComplex * b, cuComplex * c) {
  c[0] = cuCfmaf(a, b[0], c[0]); c[1] = cuCfmaf(a, b[1], c[1]);
  c[2] = cuCfmaf(a, b[2], c[2]); c[3] = cuCfmaf(a, b[3], c[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void caxpy(int n, cuComplex a, cuComplex * b, cuComplex * c) {
  c[0] = cuCfmaf(a, b[0], c[0]); if (1 >= n) return;
  c[1] = cuCfmaf(a, b[1], c[1]); if (2 >= n) return;
  c[2] = cuCfmaf(a, b[2], c[2]); if (3 >= n) return;
  c[3] = cuCfmaf(a, b[3], c[3]);
}

/**
 * CTRSM:
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
__global__ void ctrsm(int m, int n,
                      cuComplex alpha, const cuComplex * __restrict__ A, int lda,
                      cuComplex * __restrict__ B, int ldb) {

 if (side == CBlasLeft) {
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    __shared__ cuComplex a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ cuComplex b[mb][nb + 1];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    A += threadIdx.y * lda + threadIdx.x;
    B += (blockIdx.y * nb + threadIdx.y) * ldb + threadIdx.x;
    n -= blockIdx.y * nb + threadIdx.y;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) || (uplo == CBlasLower && transA != CBlasNoTrans)) {
      const int mm = m & (mb - 1);
      int i = m - mm;

      A += i * lda + i;
      cuComplex * X = B + i;

      if (mm > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[2] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConjf(A[0]);
        }

        __syncthreads();

        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        }

        __syncthreads();

        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y];
          if (2 * by < n) { X[2 * by * ldb] = b[threadIdx.x][2 * by + threadIdx.y];
          if (3 * by < n) { X[3 * by * ldb] = b[threadIdx.x][3 * by + threadIdx.y]; }}}}
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

        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[2] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        __syncthreads();

        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.x][threadIdx.y] = _A[0];
            else
              a[threadIdx.x][threadIdx.y] = cuConjf(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          caxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConjf(A[0]);
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);

        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

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
      cuComplex * X = B;
      int i = 0;

      while (m > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[2] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        __syncthreads();

        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.x][threadIdx.y] = _A[0];
            else
              a[threadIdx.x][threadIdx.y] = cuConjf(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          caxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = cuConjf(_A[0]);
        }

        __syncthreads();

        if (m < mb) break;

        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]); caxpy(3, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(2, x[1], &a[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(1, x[2], &a[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]);

        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

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

      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
      if (m > 1) { caxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
      if (m > 2) { caxpy(m - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
      if (m > 3) { caxpy(m - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); }}}}

      __syncthreads();

      b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = b[threadIdx.x][by * 0 + threadIdx.y]; if (by * 1 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 1 + threadIdx.y]; if (by * 2 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 2 + threadIdx.y]; if (by * 3 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 3 + threadIdx.y];
      }
    }
  }
  else {
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ cuComplex a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in a and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) || (uplo == CBlasLower && transA != CBlasNoTrans)) {
      cuComplex * X = B;
      int j = 0;

      while (n > 0) {
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = j;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.y][threadIdx.x] = _A[0];
            else
              a[threadIdx.y][threadIdx.x] = cuConjf(_A[0]);
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a[l], x);

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

        if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        if (n > 1) { caxpy(7, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
        if (n > 2) { caxpy(6, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
        if (n > 3) { caxpy(5, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); }}}}

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      if (n > 0) { if (diag == CBlasNonUnit)x[0] = cuCdivf(x[0], a[0][0]);
      if (n > 1) { caxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
      if (n > 2) { caxpy(n - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
      if (n > 3) { caxpy(n - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); }}}}

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuComplex * X = B + j * ldb;

      if (nn > 0) {
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.y][threadIdx.x] = A[0];
          else
            a[threadIdx.y][threadIdx.x] = cuConjf(A[0]);
        }

        __syncthreads();

        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        }

        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn) {
          X[1 * ldb] = x[1]; if (2 < nn) {
          X[2 * ldb] = x[2]; if (3 < nn) {
          X[3 * ldb] = x[3]; }}}
        }
      }

      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        __syncthreads();

        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else {
            if (transA == CBlasTrans)
              a[threadIdx.y][threadIdx.x] = _A[0];
            else
              a[threadIdx.y][threadIdx.x] = cuConjf(_A[0]);
          }

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
          caxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else {
          if (transA == CBlasTrans)
            a[threadIdx.y][threadIdx.x] = A[0];
          else
            a[threadIdx.y][threadIdx.x] = cuConjf(A[0]);
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}

template void ctrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
