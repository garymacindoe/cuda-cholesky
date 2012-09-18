#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICT__)

// y(1:4) -= alpha * x(1:4)
__device__ void caxpy(cuComplex alpha, const float * x_real, const float * x_imag, cuComplex * y) {
  y[0] = cuCsubf(y[0], cuCmulf(alpha, make_cuComplex(x_real[0], x_imag[0])));
  y[1] = cuCsubf(y[1], cuCmulf(alpha, make_cuComplex(x_real[1], x_imag[1])));
  y[2] = cuCsubf(y[2], cuCmulf(alpha, make_cuComplex(x_real[2], x_imag[2])));
  y[3] = cuCsubf(y[3], cuCmulf(alpha, make_cuComplex(x_real[3], x_imag[3])));
}

// y(1:n) -= alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const float * x_real, const float * x_imag, cuComplex * y) {
  y[0] = cuCsubf(y[0], cuCmulf(alpha, make_cuComplex(x_real[0], x_imag[0]))); if (1 >= n) return;
  y[1] = cuCsubf(y[1], cuCmulf(alpha, make_cuComplex(x_real[1], x_imag[1]))); if (2 >= n) return;
  y[2] = cuCsubf(y[2], cuCmulf(alpha, make_cuComplex(x_real[2], x_imag[2]))); if (3 >= n) return;
  y[3] = cuCsubf(y[3], cuCmulf(alpha, make_cuComplex(x_real[3], x_imag[3])));
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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ float a_real[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ float a_imag[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ float b_real[mb][nb + 1];
    __shared__ float b_imag[mb][nb + 1];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Work out the column for the current thread in A and B
    A += threadIdx.y * lda + threadIdx.x;
    B += (blockIdx.y * nb + threadIdx.y) * ldb + threadIdx.x;
    n -= blockIdx.y * nb + threadIdx.y;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start at the bottom of B and work upwards
      const int mm = m & (mb - 1);
      int i = m - mm;

      // Since we need to read B to update it we need two pointers into it: one
      // into the block we are updating (X), and one into the block we are
      // currently reading to update it (_B, defined later in terms of X).
      A += i * lda + i;
      cuComplex * X = B + i;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (mm > 0) {
        // Read the block of B we are updating and transpose into shared
        // memory using b.
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(X[j * ldb]);
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmulf(alpha, make_cuComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmulf(alpha, make_cuComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmulf(alpha, make_cuComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmulf(alpha, make_cuComplex(b_real[3][ti], b_imag[3][ti]));

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCrealf(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimagf(A[0]);
        }
        else {
          a_real[threadIdx.x][threadIdx.y] = cuCrealf(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? -cuCimagf(A[0]) : cuCimagf(A[0]);
        }

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); caxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2])); caxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1])); caxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b_real[0][ti] = cuCrealf(x[0]); b_imag[0][ti] = cuCimagf(x[0]);
        b_real[1][ti] = cuCrealf(x[1]); b_imag[1][ti] = cuCimagf(x[1]);
        b_real[2][ti] = cuCrealf(x[2]); b_imag[2][ti] = cuCimagf(x[2]);
        b_real[3][ti] = cuCrealf(x[3]); b_imag[3][ti] = cuCimagf(x[3]);

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = make_cuComplex(b_real[threadIdx.x][0 * by + threadIdx.y], b_imag[threadIdx.x][0 * by + threadIdx.y]);
          if (1 * by < n) { X[1 * by * ldb] = make_cuComplex(b_real[threadIdx.x][1 * by + threadIdx.y], b_imag[threadIdx.x][1 * by + threadIdx.y]);
          if (2 * by < n) { X[2 * by * ldb] = make_cuComplex(b_real[threadIdx.x][2 * by + threadIdx.y], b_imag[threadIdx.x][2 * by + threadIdx.y]);
          if (3 * by < n) { X[3 * by * ldb] = make_cuComplex(b_real[threadIdx.x][3 * by + threadIdx.y], b_imag[threadIdx.x][3 * by + threadIdx.y]); }}}}
        }
      }

      // Move up to the next block
      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(X[j * ldb]);
        }

        __syncthreads();

        x[0] = cuCmulf(alpha, make_cuComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmulf(alpha, make_cuComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmulf(alpha, make_cuComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmulf(alpha, make_cuComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimagf(_A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(_B[j * ldb]);
          }

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(make_cuComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(make_cuComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCrealf(A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimagf(A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCrealf(A[0]);
            a_imag[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? -cuCimagf(A[0]) : cuCimagf(A[0]);
          }

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); caxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2])); caxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1])); caxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));

        // Write X out transposing it back via shared memory using b
        b_real[0][ti] = cuCrealf(x[0]); b_imag[0][ti] = cuCimagf(x[0]);
        b_real[1][ti] = cuCrealf(x[1]); b_imag[1][ti] = cuCimagf(x[1]);
        b_real[2][ti] = cuCrealf(x[2]); b_imag[2][ti] = cuCimagf(x[2]);
        b_real[3][ti] = cuCrealf(x[3]); b_imag[3][ti] = cuCimagf(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

        // Move up to the next blocks of A and B (through X)
        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case we start at the top of B and work downwards
      cuComplex * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(X[j * ldb]);
          b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(X[j * ldb]);
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmulf(alpha, make_cuComplex(b_real[0][ti], b_imag[0][ti]));
        x[1] = cuCmulf(alpha, make_cuComplex(b_real[1][ti], b_imag[1][ti]));
        x[2] = cuCmulf(alpha, make_cuComplex(b_real[2][ti], b_imag[2][ti]));
        x[3] = cuCmulf(alpha, make_cuComplex(b_real[3][ti], b_imag[3][ti]));

        __syncthreads();

        // Start at the top of B and move down to X
        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = cuCimagf(_A[0]);
          }
          else {
            a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real[threadIdx.x][j + threadIdx.y] = cuCrealf(_B[j * ldb]);
            b_imag[threadIdx.x][j + threadIdx.y] = cuCimagf(_B[j * ldb]);
          }

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(make_cuComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(make_cuComplex(b_real[l][ti], b_imag[l][ti]), a_real[l], a_imag[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
          a_imag[threadIdx.y][threadIdx.x] = cuCimagf(_A[0]);
        }
        else {
          a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
          a_imag[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
        }

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0])); caxpy(3, x[0], &a_real[0][1], &a_imag[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1])); caxpy(2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2])); caxpy(1, x[2], &a_real[2][3], &a_imag[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3]));

        // Write X out transposing it back via shared memory using b
        b_real[0][ti] = cuCrealf(x[0]); b_imag[0][ti] = cuCimagf(x[0]);
        b_real[1][ti] = cuCrealf(x[1]); b_imag[1][ti] = cuCimagf(x[1]);
        b_real[2][ti] = cuCrealf(x[2]); b_imag[2][ti] = cuCimagf(x[2]);
        b_real[3][ti] = cuCrealf(x[3]); b_imag[3][ti] = cuCimagf(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuComplex(b_real[threadIdx.x][j + threadIdx.y], b_imag[threadIdx.x][j + threadIdx.y]);

        __syncthreads();

        // Move up to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      // Handle the trailing elements last, if any.
      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));
      if (m > 1) { caxpy(m - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1]));
      if (m > 2) { caxpy(m - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2]));
      if (m > 3) { caxpy(m - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); }}}}

      __syncthreads();

      // Write X out transposing it back via shared memory using b
      b_real[0][ti] = cuCrealf(x[0]); b_imag[0][ti] = cuCimagf(x[0]);
      b_real[1][ti] = cuCrealf(x[1]); b_imag[1][ti] = cuCimagf(x[1]);
      b_real[2][ti] = cuCrealf(x[2]); b_imag[2][ti] = cuCimagf(x[2]);
      b_real[3][ti] = cuCrealf(x[3]); b_imag[3][ti] = cuCimagf(x[3]);

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = make_cuComplex(b_real[threadIdx.x][by * 0 + threadIdx.y], b_imag[threadIdx.x][by * 0 + threadIdx.y]); if (by * 1 >= n) return; X += by * ldb;
        X[0] = make_cuComplex(b_real[threadIdx.x][by * 1 + threadIdx.y], b_imag[threadIdx.x][by * 1 + threadIdx.y]); if (by * 2 >= n) return; X += by * ldb;
        X[0] = make_cuComplex(b_real[threadIdx.x][by * 2 + threadIdx.y], b_imag[threadIdx.x][by * 2 + threadIdx.y]); if (by * 3 >= n) return; X += by * ldb;
        X[0] = make_cuComplex(b_real[threadIdx.x][by * 3 + threadIdx.y], b_imag[threadIdx.x][by * 3 + threadIdx.y]);
      }
    }
  }
  else {
   // For CBlasRight each thread updates a row.  This means that B can be read
   // efficiently straight from global memory.
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ float a_real[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ float a_imag[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      cuComplex * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        // Start at the left of B and move right to X
        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimagf(_A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
          }

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          //  Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimagf(_A[0]);
        }
        else {
          a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
          a_imag[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
        }

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));
        caxpy(3, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1]));
        caxpy(2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2]));
        caxpy(1, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3]));

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        // Move right to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      // Update X unrolled (forward loop)
      if (n > 0) {
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));
        if (n > 1) { caxpy(n - 1, x[0], &a_real[0][1], &a_imag[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1]));
        if (n > 2) { caxpy(n - 2, x[1], &a_real[1][2], &a_imag[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2]));
        if (n > 3) { caxpy(n - 3, x[2], &a_real[2][3], &a_imag[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); }}}

        // Write X
        if (ti < m) {
          X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
          X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
        }
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuComplex * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCrealf(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimagf(A[0]);
        }
        else {
          a_real[threadIdx.y][threadIdx.x] = cuCrealf(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? -cuCimagf(A[0]) : cuCimagf(A[0]);
        }

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); caxpy(3, x[3], a_real[3], a_imag[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2])); caxpy(2, x[2], a_real[2], a_imag[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1])); caxpy(1, x[1], a_real[1], a_imag[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));
        }

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn) {
          X[1 * ldb] = x[1]; if (2 < nn) {
          X[2 * ldb] = x[2]; if (3 < nn) {
          X[3 * ldb] = x[3]; }}}
        }
      }

      // Move left to the next block
      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        // Read the current block of X and multiply by alpha
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        __syncthreads();

        // Start one block beyond X and move to the right
        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans) {
            a_real[threadIdx.x][threadIdx.y] = cuCrealf(_A[0]);
            a_imag[threadIdx.x][threadIdx.y] = cuCimagf(_A[0]);
          }
          else {
            a_real[threadIdx.y][threadIdx.x] = cuCrealf(_A[0]);
            a_imag[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? -cuCimagf(_A[0]) : cuCimagf(_A[0]);
          }

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a_real[l], a_imag[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(_B[l * ldb], a_real[l], a_imag[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real[threadIdx.x][threadIdx.y] = cuCrealf(A[0]);
          a_imag[threadIdx.x][threadIdx.y] = cuCimagf(A[0]);
        }
        else {
          a_real[threadIdx.y][threadIdx.x] = cuCrealf(A[0]);
          a_imag[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? -cuCimagf(A[0]) : cuCimagf(A[0]);
        }

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit) x[3] = cuCmulf(x[3], make_cuComplex(a_real[3][3], a_imag[3][3])); caxpy(3, x[3], a_real[3], a_imag[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCmulf(x[2], make_cuComplex(a_real[2][2], a_imag[2][2])); caxpy(2, x[2], a_real[2], a_imag[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCmulf(x[1], make_cuComplex(a_real[1][1], a_imag[1][1])); caxpy(1, x[1], a_real[1], a_imag[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCmulf(x[0], make_cuComplex(a_real[0][0], a_imag[0][0]));

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        // Move left to the next blocks of A and B (through X)
        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}

#else

// y(1:4) -= alpha * x(1:4)
__device__ void caxpy(cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCsubf(y[0], cuCmulf(alpha, x[0]));
  y[1] = cuCsubf(y[1], cuCmulf(alpha, x[1]));
  y[2] = cuCsubf(y[2], cuCmulf(alpha, x[2]));
  y[3] = cuCsubf(y[3], cuCmulf(alpha, x[3]));
}

// y(1:n) -= alpha * x(1:n)
__device__ void caxpy(int n, cuComplex alpha, const cuComplex * x, cuComplex * y) {
  y[0] = cuCsubf(y[0], cuCmulf(alpha, x[0])); if (1 >= n) return;
  y[1] = cuCsubf(y[1], cuCmulf(alpha, x[1])); if (2 >= n) return;
  y[2] = cuCsubf(y[2], cuCmulf(alpha, x[2])); if (3 >= n) return;
  y[3] = cuCsubf(y[3], cuCmulf(alpha, x[3]));
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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ cuComplex a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ cuComplex b[mb][nb + 1];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Work out the column for the current thread in A and B
    A += threadIdx.y * lda + threadIdx.x;
    B += (blockIdx.y * nb + threadIdx.y) * ldb + threadIdx.x;
    n -= blockIdx.y * nb + threadIdx.y;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start at the bottom of B and work upwards
      const int mm = m & (mb - 1);
      int i = m - mm;

      // Since we need to read B to update it we need two pointers into it: one
      // into the block we are updating (X), and one into the block we are
      // currently reading to update it (_B, defined later in terms of X).
      A += i * lda + i;
      cuComplex * X = B + i;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (mm > 0) {
        // Read the block of B we are updating and transpose into shared
        // memory using b.
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[1] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConjf(A[0]) : A[0];

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y];
          if (2 * by < n) { X[2 * by * ldb] = b[threadIdx.x][2 * by + threadIdx.y];
          if (3 * by < n) { X[3 * by * ldb] = b[threadIdx.x][3 * by + threadIdx.y]; }}}}
        }
      }

      // Move up to the next block
      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[2] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConjf(A[0]) : A[0];

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit) x[3] = cuCmulf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCmulf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCmulf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCmulf(x[0], a[0][0]);

        // Write X out transposing it back via shared memory using b
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];

        // Move up to the next blocks of A and B (through X)
        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case we start at the top of B and work downwards
      cuComplex * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmulf(alpha, b[0][ti]); x[1] = cuCmulf(alpha, b[1][ti]);
        x[2] = cuCmulf(alpha, b[2][ti]); x[3] = cuCmulf(alpha, b[3][ti]);

        __syncthreads();

        // Start at the top of B and move down to X
        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            caxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]); caxpy(3, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(2, x[1], &a[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(1, x[2], &a[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]);

        // Write X out transposing it back via shared memory using b
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = b[threadIdx.x][j + threadIdx.y];

        __syncthreads();

        // Move up to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      // Handle the trailing elements last, if any.
      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
      if (m > 1) { caxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
      if (m > 2) { caxpy(m - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
      if (m > 3) { caxpy(m - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); }}}}

      __syncthreads();

      // Write X out transposing it back via shared memory using b
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
   // For CBlasRight each thread updates a row.  This means that B can be read
   // efficiently straight from global memory.
//     typedef char _x[(nb == 4) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ cuComplex a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuComplex x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      cuComplex * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        // Start at the left of B and move right to X
        const cuComplex * _A = A;
        const cuComplex * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          //  Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = _A[0];
        else
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        caxpy(3, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
        caxpy(2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
        caxpy(1, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]);

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        __syncthreads();

        // Move right to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      // Update X unrolled (forward loop)
      if (n > 0) {
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        if (n > 1) { caxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]);
        if (n > 2) { caxpy(n - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]);
        if (n > 3) { caxpy(n - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); }}}

        // Write X
        if (ti < m) {
          X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
          X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
        }
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuComplex * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConjf(A[0]) : A[0];

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);
        }

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn) {
          X[1 * ldb] = x[1]; if (2 < nn) {
          X[2 * ldb] = x[2]; if (3 < nn) {
          X[3 * ldb] = x[3]; }}}
        }
      }

      // Move left to the next block
      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        // Read the current block of X and multiply by alpha
        x[0] = cuCmulf(alpha, X[0 * ldb]); x[1] = cuCmulf(alpha, X[1 * ldb]);
        x[2] = cuCmulf(alpha, X[2 * ldb]); x[3] = cuCmulf(alpha, X[3 * ldb]);

        __syncthreads();

        // Start one block beyond X and move to the right
        const cuComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConjf(_A[0]) : _A[0];

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            caxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          caxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConjf(A[0]) : A[0];

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit) x[3] = cuCdivf(x[3], a[3][3]); caxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] = cuCdivf(x[2], a[2][2]); caxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] = cuCdivf(x[1], a[1][1]); caxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdivf(x[0], a[0][0]);

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
        }

        // Move left to the next blocks of A and B (through X)
        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}
#endif

template void ctrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasTrans,     CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasTrans,     CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasConjTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasUpper, CBlasConjTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasNoTrans,   CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasNoTrans,   CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasTrans,     CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasTrans,     CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasConjTrans, CBlasNonUnit,  4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasLeft,  CBlasLower, CBlasConjTrans, CBlasUnit,     4, 16, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasNoTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasNoTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasTrans,     CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasTrans,     CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasNoTrans,   CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasNoTrans,   CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasTrans,     CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasTrans,     CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, 16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
template void ctrsm<CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit,    16,  4, 4, 4>(int, int, cuComplex, const cuComplex * __restrict__, int, cuComplex * __restrict__, int);
