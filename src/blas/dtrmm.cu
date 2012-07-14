#include "blas.h"

#if __CUDA_ARCH__ < 200
// y(1:4) += alpha * x(1:4)
__device__ void daxpy(double a, const int * b_lo, const int * b_hi, double * c) {
  c[0] -= a * __hiloint2double(b_hi[0], b_lo[0]);
  c[1] -= a * __hiloint2double(b_hi[1], b_lo[1]);
  c[2] -= a * __hiloint2double(b_hi[2], b_lo[2]);
  c[3] -= a * __hiloint2double(b_hi[3], b_lo[3]);
}

// y(1:n) += alpha * x(1:n)
__device__ void daxpy(int n, double a, const int * b_lo, const int * b_hi, double * c) {
  c[0] -= a * __hiloint2double(b_hi[0], b_lo[0]); if (1 >= n) return;
  c[1] -= a * __hiloint2double(b_hi[1], b_lo[1]); if (2 >= n) return;
  c[2] -= a * __hiloint2double(b_hi[2], b_lo[2]); if (3 >= n) return;
  c[3] -= a * __hiloint2double(b_hi[3], b_lo[3]);
}
#else
// y(1:4) += alpha * x(1:4)
__device__ void daxpy(double a, double * b, double * c) {
  c[0] -= a * b[0]; c[1] -= a * b[1]; c[2] -= a * b[2]; c[3] -= a * b[3];
}

// y(1:n) += alpha * x(1:n)
__device__ void daxpy(int n, double a, double * b, double * c) {
  c[0] -= a * b[0]; if (1 >= n) return; c[1] -= a * b[1]; if (2 >= n) return;
  c[2] -= a * b[2]; if (3 >= n) return; c[3] -= a * b[3];
}
#endif

/**
 * DTRSM:
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
__global__ void dtrsm(int m, int n,
                      double alpha, const double * __restrict__ A, int lda,
                      double * __restrict__ B, int ldb) {

 if (side == CBlasLeft) {
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

#if __CUDA_ARCH__ < 200
    __shared__ int a_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int b_lo[mb][nb + 1];
    __shared__ int b_hi[mb][nb + 1];
#else
    __shared__ double a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ double b[mb][nb + 1];
#endif
    double x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    A += threadIdx.y * lda + threadIdx.x;
    B += (blockIdx.y * nb + threadIdx.y) * ldb + threadIdx.x;
    n -= blockIdx.y * nb + threadIdx.y;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) || (uplo == CBlasLower && transA != CBlasNoTrans)) {
      const int mm = m & (mb - 1);
      int i = m - mm;

      A += i * lda + i;
      double * X = B + i;

      if (mm > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
#if __CUDA_ARCH__ < 200
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
#else
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];
#endif
        }

        __syncthreads();

#if __CUDA_ARCH__ < 200
        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);
#else
        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];
#endif

        if (transA == CBlasNoTrans) {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
#else
          a[threadIdx.y][threadIdx.x] = A[0];
#endif
        }
        else {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
#else
          a[threadIdx.x][threadIdx.y] = A[0];
#endif
        }

        __syncthreads();

        switch (mm - 1) {
#if __CUDA_ARCH__ < 200
          case 3: if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
#else
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
#endif
        }

        __syncthreads();

#if __CUDA_ARCH__ < 200
        b_lo[0][ti] = __double2loint(x[0]); b_hi[0][ti] = __double2hiint(x[0]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);
#else
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
#endif

        __syncthreads();

        if (threadIdx.x < mm) {
#if __CUDA_ARCH__ < 200
          if (0 * by < n) { X[0 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][0 * by + threadIdx.y], b_lo[threadIdx.x][0 * by + threadIdx.y]);
          if (1 * by < n) { X[1 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][1 * by + threadIdx.y], b_lo[threadIdx.x][1 * by + threadIdx.y]);
          if (2 * by < n) { X[2 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][2 * by + threadIdx.y], b_lo[threadIdx.x][2 * by + threadIdx.y]);
          if (3 * by < n) { X[3 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][3 * by + threadIdx.y], b_lo[threadIdx.x][3 * by + threadIdx.y]); }}}}
#else
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y];
          if (2 * by < n) { X[2 * by * ldb] = b[threadIdx.x][2 * by + threadIdx.y];
          if (3 * by < n) { X[3 * by * ldb] = b[threadIdx.x][3 * by + threadIdx.y]; }}}}
#endif
        }
      }

      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by) {
#if __CUDA_ARCH__ < 200
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
#else
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];
#endif
        }

        __syncthreads();

#if __CUDA_ARCH__ < 200
        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);
#else
        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];
#endif

        __syncthreads();

        const double * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const double * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

        if (transA == CBlasNoTrans) {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
#else
          a[threadIdx.y][threadIdx.x] = _A[0];
#endif
        }
        else {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
#else
          a[threadIdx.x][threadIdx.y] = _A[0];
#endif
        }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
#if __CUDA_ARCH__ < 200
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(_B[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(_B[j * ldb]);
#else
          b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];
#endif
        }

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
#if __CUDA_ARCH__ < 200
            daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);
#else
            daxpy(b[l][ti], a[l], x);
#endif

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
#if __CUDA_ARCH__ < 200
          daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);
#else
          daxpy(b[l][ti], a[l], x);
#endif

        __syncthreads();

        if (transA == CBlasNoTrans) {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
#else
          a[threadIdx.y][threadIdx.x] = A[0];
#endif
        }
        else {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
#else
          a[threadIdx.x][threadIdx.y] = A[0];
#endif
        }

        __syncthreads();

#if __CUDA_ARCH__ < 200
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
#else
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];
#endif

#if __CUDA_ARCH__ < 200
        b_lo[0][ti] = __double2loint(x[0]); b_hi[0][ti] = __double2hiint(x[0]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);
#else
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
#endif

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
#if __CUDA_ARCH__ < 200
          X[j * ldb] = __hiloint2double(b_hi[threadIdx.x][j + threadIdx.y], b_lo[threadIdx.x][j + threadIdx.y]);
#else
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];
#endif

        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      double * X = B;
      int i = 0;

      while (m > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
#if __CUDA_ARCH__ < 200
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
#else
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];
#endif
        }

        __syncthreads();

#if __CUDA_ARCH__ < 200
        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);
#else
        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];
#endif

        __syncthreads();

        const double * _A = A;
        const double * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans) {
#if __CUDA_ARCH__ < 200
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
#else
            a[threadIdx.y][threadIdx.x] = _A[0];
#endif
          }
          else {
#if __CUDA_ARCH__ < 200
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
#else
            a[threadIdx.x][threadIdx.y] = _A[0];
#endif
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
#if __CUDA_ARCH__ < 200
            b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(_B[j * ldb]);
            b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(_B[j * ldb]);
#else
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];
#endif
          }

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
#if __CUDA_ARCH__ < 200
            daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);
#else
            daxpy(b[l][ti], a[l], x);
#endif

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
#if __CUDA_ARCH__ < 200
          daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);
#else
          daxpy(b[l][ti], a[l], x);
#endif

        __syncthreads();

        if (transA == CBlasNoTrans) {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
#else
          a[threadIdx.y][threadIdx.x] = _A[0];
#endif
        }
        else {
#if __CUDA_ARCH__ < 200
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
#else
          a[threadIdx.x][threadIdx.y] = _A[0];
#endif
        }

        __syncthreads();

        if (m < mb) break;

#if __CUDA_ARCH__ < 200
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]); daxpy(3, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(1, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]);
#else
        if (diag == CBlasNonUnit) x[0] /= a[0][0]; daxpy(3, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(2, x[1], &a[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(1, x[2], &a[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] /= a[3][3];
#endif

#if __CUDA_ARCH__ < 200
        b_lo[0][ti] = __double2loint(x[0]); b_hi[0][ti] = __double2hiint(x[0]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);
#else
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
#endif

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
#if __CUDA_ARCH__ < 200
          X[j * ldb] = __hiloint2double(b_hi[threadIdx.x][j + threadIdx.y], b_lo[threadIdx.x][j + threadIdx.y]);
#else
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];
#endif

        __syncthreads();

        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

#if __CUDA_ARCH__ < 200
      if (m > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_hi[0][0]);
      if (m > 1) { daxpy(m - 1, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_hi[1][1]);
      if (m > 2) { daxpy(m - 2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_hi[2][2]);
      if (m > 3) { daxpy(m - 3, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_hi[3][3]); }}}}
#else
      if (m > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (m > 1) { daxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (m > 2) { daxpy(m - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (m > 3) { daxpy(m - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}
#endif

      __syncthreads();

#if __CUDA_ARCH__ < 200
      b_lo[0][ti] = __double2loint(x[0]); b_hi[0][ti] = __double2hiint(x[0]);
      b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
      b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
      b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);
#else
      b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
#endif

      __syncthreads();

      if (threadIdx.x < m) {
#if __CUDA_ARCH__ < 200
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 0 + threadIdx.y], b_lo[threadIdx.x][by * 0 + threadIdx.y]); if (by * 1 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 1 + threadIdx.y], b_lo[threadIdx.x][by * 1 + threadIdx.y]); if (by * 2 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 2 + threadIdx.y], b_lo[threadIdx.x][by * 2 + threadIdx.y]); if (by * 3 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 3 + threadIdx.y], b_lo[threadIdx.x][by * 3 + threadIdx.y]);
#else
        X[0] = b[threadIdx.x][by * 0 + threadIdx.y]; if (by * 1 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 1 + threadIdx.y]; if (by * 2 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 2 + threadIdx.y]; if (by * 3 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 3 + threadIdx.y];
#endif
      }
    }
  }
  else {
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
#if __CUDA_ARCH__ < 200
    __shared__ int a_lo[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ int a_hi[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
#else
    __shared__ double a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
#endif
    double x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in a and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      double * X = B;
      int j = 0;

      while (n > 0) {
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb];
        x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        const double * _A = A;
        const double * _B = B;
        int k = j;
        while (k > 0) {

#if __CUDA_ARCH__ < 200
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
          }
          else {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
          }
#else
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = _A[0];
#endif

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
#if __CUDA_ARCH__ < 200
            daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);
#else
            daxpy(_B[l * ldb], a[l], x);
#endif

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

#if __CUDA_ARCH__ < 200
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
        }
#else
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = _A[0];
        else
          a[threadIdx.y][threadIdx.x] = _A[0];
#endif

        __syncthreads();

        if (n < nb) break;

#if __CUDA_ARCH__ < 200
        if (n > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
        if (n > 1) { daxpy(3, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]);
        if (n > 2) { daxpy(2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]);
        if (n > 3) { daxpy(1, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); }}}}
#else
        if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
        if (n > 1) { daxpy(3, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
        if (n > 2) { daxpy(2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
        if (n > 3) { daxpy(1, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}
#endif

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
          X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

#if __CUDA_ARCH__ < 200
      if (n > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
      if (n > 1) { daxpy(3, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]);
      if (n > 2) { daxpy(2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]);
      if (n > 3) { daxpy(1, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); }}}}
#else
      if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (n > 1) { daxpy(3, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (n > 2) { daxpy(2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (n > 3) { daxpy(1, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}
#endif

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      double * X = B + j * ldb;

      if (nn > 0) {
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb];
        x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

#if __CUDA_ARCH__ < 200
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
        }
#else
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];
#endif

        __syncthreads();

        switch (nn - 1) {
#if __CUDA_ARCH__ < 200
          case 3: if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_hi[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_hi[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_hi[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_hi[0][0]);
#else
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
#endif
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
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb];
        x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        __syncthreads();

        const double * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const double * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

#if __CUDA_ARCH__ < 200
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
        }
#else
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = _A[0];
        else
          a[threadIdx.y][threadIdx.x] = _A[0];
#endif

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
#if __CUDA_ARCH__ < 200
            daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);
#else
            daxpy(_B[l * ldb], a[l], x);
#endif

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
#if __CUDA_ARCH__ < 200
          daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);
#else
          daxpy(_B[l * ldb], a[l], x);
#endif

        __syncthreads();

#if __CUDA_ARCH__ < 200
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
        }
#else
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];
#endif

        __syncthreads();

#if __CUDA_ARCH__ < 200
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
#else
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];
#endif

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
          X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}

template void dtrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,     4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
template void dtrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, double, const double * __restrict__, int, double * __restrict__, int);
