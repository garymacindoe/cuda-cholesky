#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 || defined(__BANK_CONFLICT__)

// y(1:4) += alpha * x(1:4)
__device__ void zaxpy(cuDoubleComplex a, int * b_real_lo, int * b_real_hi,
                      int * b_imag_lo, int * b_imag_hi, cuDoubleComplex * c) {
  c[0] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[0], b_real_lo[0]),
                     __hiloint2double(b_imag_hi[0], b_imag_lo[0])), c[0]);

  c[1] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[1], b_real_lo[1]),
                     __hiloint2double(b_imag_hi[1], b_imag_lo[1])), c[1]);

  c[2] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[2], b_real_lo[2]),
                     __hiloint2double(b_imag_hi[2], b_imag_lo[2])), c[2]);

  c[3] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[3], b_real_lo[3]),
                     __hiloint2double(b_imag_hi[3], b_imag_lo[3])), c[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex a, int * b_real_lo, int * b_real_hi,
                      int * b_imag_lo, int * b_imag_hi, cuDoubleComplex * c) {
  c[0] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[0], b_real_lo[0]),
                     __hiloint2double(b_imag_hi[0], b_imag_lo[0])), c[0]);
  if (1 >= n) return;
  c[1] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[1], b_real_lo[1]),
                     __hiloint2double(b_imag_hi[1], b_imag_lo[1])), c[1]);
  if (2 >= n) return;
  c[2] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[2], b_real_lo[2]),
                     __hiloint2double(b_imag_hi[2], b_imag_lo[2])), c[2]);
  if (3 >= n) return;
  c[3] = cuCfma(a, make_cuDoubleComplex(
                     __hiloint2double(b_real_hi[3], b_real_lo[3]),
                     __hiloint2double(b_imag_hi[3], b_imag_lo[3])), c[3]);
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
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    __shared__ int a_real_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_real_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_imag_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_imag_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int b_real_lo[mb][nb + 1];
    __shared__ int b_real_hi[mb][nb + 1];
    __shared__ int b_imag_lo[mb][nb + 1];
    __shared__ int b_imag_hi[mb][nb + 1];

    cuDoubleComplex x[4];

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
        for (int j = 0; j < nb; j += by) {
          b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(X[j * ldb]));
          b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(X[j * ldb]));
          b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(X[j * ldb]));
          b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(X[j * ldb]));
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[0][ti], b_real_lo[0][ti]), __hiloint2double(b_imag_hi[0][ti], b_imag_lo[0][ti])));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[1][ti], b_real_lo[1][ti]), __hiloint2double(b_imag_hi[1][ti], b_imag_lo[1][ti])));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[2][ti], b_real_lo[2][ti]), __hiloint2double(b_imag_hi[2][ti], b_imag_lo[2][ti])));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[3][ti], b_real_lo[3][ti]), __hiloint2double(b_imag_hi[3][ti], b_imag_lo[3][ti])));
//TODO: here
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
        }

        __syncthreads();

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][0 * by + threadIdx.y], b_imag[threadIdx.x][0 * by + threadIdx.y]);
          if (1 * by < n) { X[1 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][1 * by + threadIdx.y], b_imag[threadIdx.x][1 * by + threadIdx.y]);
          if (2 * by < n) { X[2 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][2 * by + threadIdx.y], b_imag[threadIdx.x][2 * by + threadIdx.y]);
          if (3 * by < n) { X[3 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][3 * by + threadIdx.y], b_imag[threadIdx.x][3 * by + threadIdx.y]); }}}}
        }
      }

      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCreal(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuDoubleComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = -cuCimag(_A[0]);
            }
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCreal(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(_B[j * ldb]);
          }

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

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
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCreal(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = -cuCimag(_A[0]);
            }
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCreal(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(_B[j * ldb]);
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (m < mb) break;

        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0])); zaxpy(3, x[0], &a_real[0][1], &a_imag[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(1, x[2], &a_real[2][3], &a_imag[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3]));

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

        __syncthreads();

        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
      if (m > 1) { zaxpy(m - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
      if (m > 2) { zaxpy(m - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
      if (m > 3) { zaxpy(m - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

      __syncthreads();

      b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
      b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
      b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
      b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = make_cuDoubleComplex(b_real[threadIdx.x][by * 0 + threadIdx.y], b_imag[threadIdx.x][by * 0 + threadIdx.y]); if (by * 1 >= n) return; X += by * ldb;
        X[1] = make_cuDoubleComplex(b_real[threadIdx.x][by * 1 + threadIdx.y], b_imag[threadIdx.x][by * 1 + threadIdx.y]); if (by * 2 >= n) return; X += by * ldb;
        X[2] = make_cuDoubleComplex(b_real[threadIdx.x][by * 2 + threadIdx.y], b_imag[threadIdx.x][by * 2 + threadIdx.y]); if (by * 3 >= n) return; X += by * ldb;
        X[3] = make_cuDoubleComplex(b_real[threadIdx.x][by * 3 + threadIdx.y], b_imag[threadIdx.x][by * 3 + threadIdx.y]);
      }
    }
  }
  else {
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ double a_real[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ double a_imag[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];

    cuDoubleComplex x[4];

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
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = j;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
            }
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
          }
        }

        __syncthreads();

        if (n < nb) break;

        if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
        if (n > 1) { zaxpy(7, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
        if (n > 2) { zaxpy(6, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
        if (n > 3) { zaxpy(5, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      if (n > 0) { if (diag == CBlasNonUnit)x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
      if (n > 1) { zaxpy(n - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
      if (n > 2) { zaxpy(n - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
      if (n > 3) { zaxpy(n - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuDoubleComplex * X = B + j * ldb;

      if (nn > 0) {
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
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
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuDoubleComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
            }
          }

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));

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

#else

// y(1:8) += alpha * x(1:8)
__device__ void zaxpy(cuDoubleComplex a, double * b_real, double * b_imag, cuDoubleComplex * c) {
  c[0] = cuCfma(a, make_cuDoubleComplex(b_real[0], b_imag[0]), c[0]);
  c[1] = cuCfma(a, make_cuDoubleComplex(b_real[1], b_imag[1]), c[1]);
  c[2] = cuCfma(a, make_cuDoubleComplex(b_real[2], b_imag[2]), c[2]);
  c[3] = cuCfma(a, make_cuDoubleComplex(b_real[3], b_imag[3]), c[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex a, double * b_real, double * b_imag, cuDoubleComplex * c) {
  c[0] = cuCfma(a, make_cuDoubleComplex(b_real[0], b_imag[0]), c[0]); if (1 >= n) return;
  c[1] = cuCfma(a, make_cuDoubleComplex(b_real[1], b_imag[1]), c[1]); if (2 >= n) return;
  c[2] = cuCfma(a, make_cuDoubleComplex(b_real[2], b_imag[2]), c[2]); if (3 >= n) return;
  c[3] = cuCfma(a, make_cuDoubleComplex(b_real[3], b_imag[3]), c[3]);
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
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    __shared__ double a_real[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ double a_imag[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ double b_real[mb][nb + 1];
    __shared__ double b_imag[mb][nb + 1];

    cuDoubleComplex x[4];

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
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCreal(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(b_real[3][ti], b_imag[3][ti]));

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
        }

        __syncthreads();

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][0 * by + threadIdx.y], b_imag[threadIdx.x][0 * by + threadIdx.y]);
          if (1 * by < n) { X[1 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][1 * by + threadIdx.y], b_imag[threadIdx.x][1 * by + threadIdx.y]);
          if (2 * by < n) { X[2 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][2 * by + threadIdx.y], b_imag[threadIdx.x][2 * by + threadIdx.y]);
          if (3 * by < n) { X[3 * by * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][3 * by + threadIdx.y], b_imag[threadIdx.x][3 * by + threadIdx.y]); }}}}
        }
      }

      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCreal(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuDoubleComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = -cuCimag(_A[0]);
            }
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCreal(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(_B[j * ldb]);
          }

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

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
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCreal(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmul(alpha, make_cuDoubleComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmul(alpha, make_cuDoubleComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmul(alpha, make_cuDoubleComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
              a_imag[threadIdx.x][threadIdx.y] = -cuCimag(_A[0]);
            }
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCreal(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimag(_B[j * ldb]);
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (m < mb) break;

        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0])); zaxpy(3, x[0], &a_real[0][1], &a_imag[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(1, x[2], &a_real[2][3], &a_imag[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3]));

        b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
        b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
        b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
        b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

        __syncthreads();

        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
      if (m > 1) { zaxpy(m - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
      if (m > 2) { zaxpy(m - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
      if (m > 3) { zaxpy(m - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

      __syncthreads();

      b_real[0][ti] = cuCreal(x[0]); b_imag[0][ti] = cuCimag(x[0]);
      b_real[1][ti] = cuCreal(x[1]); b_imag[1][ti] = cuCimag(x[1]);
      b_real[2][ti] = cuCreal(x[2]); b_imag[2][ti] = cuCimag(x[2]);
      b_real[3][ti] = cuCreal(x[3]); b_imag[3][ti] = cuCimag(x[3]);

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = make_cuDoubleComplex(b_real[threadIdx.x][by * 0 + threadIdx.y], b_imag[threadIdx.x][by * 0 + threadIdx.y]); if (by * 1 >= n) return; X += by * ldb;
        X[1] = make_cuDoubleComplex(b_real[threadIdx.x][by * 1 + threadIdx.y], b_imag[threadIdx.x][by * 1 + threadIdx.y]); if (by * 2 >= n) return; X += by * ldb;
        X[2] = make_cuDoubleComplex(b_real[threadIdx.x][by * 2 + threadIdx.y], b_imag[threadIdx.x][by * 2 + threadIdx.y]); if (by * 3 >= n) return; X += by * ldb;
        X[3] = make_cuDoubleComplex(b_real[threadIdx.x][by * 3 + threadIdx.y], b_imag[threadIdx.x][by * 3 + threadIdx.y]);
      }
    }
  }
  else {
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ double a_real[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ double a_imag[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];

    cuDoubleComplex x[4];

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
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = j;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
            }
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
          }
        }

        __syncthreads();

        if (n < nb) break;

        if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
        if (n > 1) { zaxpy(7, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
        if (n > 2) { zaxpy(6, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
        if (n > 3) { zaxpy(5, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      if (n > 0) { if (diag == CBlasNonUnit)x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
      if (n > 1) { zaxpy(n - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1]));
      if (n > 2) { zaxpy(n - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2]));
      if (n > 3) { zaxpy(n - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); }}}}

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuDoubleComplex * X = B + j * ldb;

      if (nn > 0) {
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));
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
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);
        x[2] = cuCmul(alpha, X[2 * ldb]); x[3] = cuCmul(alpha, X[3 * ldb]);

        __syncthreads();

        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuDoubleComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCreal(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimag(_A[0]);
          }
          else {
            if (transA == CBlasTrans) {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = cuCimag(_A[0]);
            }
            else {
              a_real[threadIdx.y][threadIdx.x] = cuCreal(_A[0]);
              a_imag[threadIdx.y][threadIdx.x] = -cuCimag(_A[0]);
            }
          }

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
          zaxpy(_B[l * ldb], a_real[l], a_imag[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCreal(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimag(A[0]);
        }
        else {
          if (transA == CBlasTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimag(A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCreal(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = -cuCimag(A[0]);
          }
        }

        __syncthreads();

        if (diag == CBlasNonUnit) x[3] = cuCdiv(x[3], make_cuDoubleComplex(a_real[3][3], a_imag[3][3])); zaxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdiv(x[2], make_cuDoubleComplex(a_real[2][2], a_imag[2][2])); zaxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], make_cuDoubleComplex(a_real[1][1], a_imag[1][1])); zaxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], make_cuDoubleComplex(a_real[0][0], a_imag[0][0]));

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

#endif

template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
