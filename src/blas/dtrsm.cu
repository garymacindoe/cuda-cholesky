#include "blas.h"

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICT__)

// y(1:4) -= alpha * x(1:4)
__device__ void daxpy(double alpha, const int * x_lo, const int * x_hi, double * y) {
  y[0] -= alpha * __hiloint2double(x_hi[0], x_lo[0]);
  y[1] -= alpha * __hiloint2double(x_hi[1], x_lo[1]);
  y[2] -= alpha * __hiloint2double(x_hi[2], x_lo[2]);
  y[3] -= alpha * __hiloint2double(x_hi[3], x_lo[3]);
}

// y(1:n) -= alpha * x(1:n)
__device__ void daxpy(int n, double alpha, const int * x_lo, const int * x_hi, double * y) {
  y[0] -= alpha * __hiloint2double(x_hi[0], x_lo[0]); if (1 >= n) return;
  y[1] -= alpha * __hiloint2double(x_hi[1], x_lo[1]); if (2 >= n) return;
  y[2] -= alpha * __hiloint2double(x_hi[2], x_lo[2]); if (3 >= n) return;
  y[3] -= alpha * __hiloint2double(x_hi[3], x_lo[3]);
}

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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ int a_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int b_lo[mb][nb + 1];
    __shared__ int b_hi[mb][nb + 1];
    double x[4];

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
      double * X = B + i;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (mm > 0) {
        // Read the block of B we are updating and transpose into shared
        // memory using b.
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
        }
        else {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
        }

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b_lo[0][ti] = __double2loint(x[0]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][0 * by + threadIdx.y], b_lo[threadIdx.x][0 * by + threadIdx.y]);
          if (1 * by < n) { X[1 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][1 * by + threadIdx.y], b_lo[threadIdx.x][1 * by + threadIdx.y]);
          if (2 * by < n) { X[2 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][2 * by + threadIdx.y], b_lo[threadIdx.x][2 * by + threadIdx.y]);
          if (3 * by < n) { X[3 * by * ldb] = __hiloint2double(b_hi[threadIdx.x][3 * by + threadIdx.y], b_lo[threadIdx.x][3 * by + threadIdx.y]); }}}}
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
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
        }

        __syncthreads();

        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const double * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const double * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
          }
          else {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(_B[j * ldb]);
            b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(_B[j * ldb]);
          }

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
          }
          else {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
          }

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);

        // Write X out transposing it back via shared memory using b
        b_lo[0][ti] = __double2loint(x[0]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = __hiloint2double(b_hi[threadIdx.x][j + threadIdx.y], b_lo[threadIdx.x][j + threadIdx.y]);

        // Move up to the next blocks of A and B (through X)
        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case we start at the top of B and work downwards
      double * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(X[j * ldb]);
          b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(X[j * ldb]);
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = alpha * __hiloint2double(b_hi[0][ti], b_lo[0][ti]);
        x[1] = alpha * __hiloint2double(b_hi[1][ti], b_lo[1][ti]);
        x[2] = alpha * __hiloint2double(b_hi[2][ti], b_lo[2][ti]);
        x[3] = alpha * __hiloint2double(b_hi[3][ti], b_lo[3][ti]);

        __syncthreads();

        // Start at the top of B and move down to X
        const double * _A = A;
        const double * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
          }
          else {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_lo[threadIdx.x][j + threadIdx.y] = __double2loint(_B[j * ldb]);
            b_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(_B[j * ldb]);
          }

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(__hiloint2double(b_hi[l][ti], b_lo[l][ti]), a_lo[l], a_hi[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
        }
        else {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
        }

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]); daxpy(3, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(1, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]);

        // Write X out transposing it back via shared memory using b
        b_lo[0][ti] = __double2loint(x[0]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = __hiloint2double(b_hi[threadIdx.x][j + threadIdx.y], b_lo[threadIdx.x][j + threadIdx.y]);

        __syncthreads();

        // Move up to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      // Handle the trailing elements last, if any.
      if (m > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
      if (m > 1) { daxpy(m - 1, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]);
      if (m > 2) { daxpy(m - 2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]);
      if (m > 3) { daxpy(m - 3, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); }}}}

      __syncthreads();

      // Write X out transposing it back via shared memory using b
        b_lo[0][ti] = __double2loint(x[0]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[1][ti] = __double2loint(x[1]); b_hi[1][ti] = __double2hiint(x[1]);
        b_lo[2][ti] = __double2loint(x[2]); b_hi[2][ti] = __double2hiint(x[2]);
        b_lo[3][ti] = __double2loint(x[3]); b_hi[3][ti] = __double2hiint(x[3]);

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 0 + threadIdx.y], b_lo[threadIdx.x][by * 0 + threadIdx.y]); if (by * 1 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 1 + threadIdx.y], b_lo[threadIdx.x][by * 1 + threadIdx.y]); if (by * 2 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 2 + threadIdx.y], b_lo[threadIdx.x][by * 2 + threadIdx.y]); if (by * 3 >= n) return; X += by * ldb;
        X[0] = __hiloint2double(b_hi[threadIdx.x][by * 3 + threadIdx.y], b_lo[threadIdx.x][by * 3 + threadIdx.y]);
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
    __shared__ int a_lo[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ int a_hi[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    double x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      double * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        // Start at the left of B and move right to X
        const double * _A = A;
        const double * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
          }
          else {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
          }

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);

          __syncthreads();

          //  Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
        }

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (n > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
        if (n > 1) { daxpy(3, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]);
        if (n > 2) { daxpy(2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]);
        if (n > 3) { daxpy(1, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); }}}}

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
      if (n > 0) { if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
      if (n > 1) { daxpy(n - 1, x[0], &a_lo[0][1], &a_hi[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]);
      if (n > 2) { daxpy(n - 2, x[1], &a_lo[1][2], &a_hi[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]);
      if (n > 3) { daxpy(n - 3, x[2], &a_lo[2][3], &a_hi[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); }}}}

      // Write X
      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      double * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
        }

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);
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
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        __syncthreads();

        // Start one block beyond X and move to the right
        const double * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const double * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans) {
            a_lo[threadIdx.x][threadIdx.y] = __double2loint(_A[0]);
            a_hi[threadIdx.x][threadIdx.y] = __double2hiint(_A[0]);
          }
          else {
            a_lo[threadIdx.y][threadIdx.x] = __double2loint(_A[0]);
            a_hi[threadIdx.y][threadIdx.x] = __double2hiint(_A[0]);
          }

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(_B[l * ldb], a_lo[l], a_hi[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_lo[threadIdx.x][threadIdx.y] = __double2loint(A[0]);
          a_hi[threadIdx.x][threadIdx.y] = __double2hiint(A[0]);
        }
        else {
          a_lo[threadIdx.y][threadIdx.x] = __double2loint(A[0]);
          a_hi[threadIdx.y][threadIdx.x] = __double2hiint(A[0]);
        }

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit) x[3] /= __hiloint2double(a_hi[3][3], a_lo[3][3]); daxpy(3, x[3], a_lo[3], a_hi[3], x);
        if (diag == CBlasNonUnit) x[2] /= __hiloint2double(a_hi[2][2], a_lo[2][2]); daxpy(2, x[2], a_lo[2], a_hi[2], x);
        if (diag == CBlasNonUnit) x[1] /= __hiloint2double(a_hi[1][1], a_lo[1][1]); daxpy(1, x[1], a_lo[1], a_hi[1], x);
        if (diag == CBlasNonUnit) x[0] /= __hiloint2double(a_hi[0][0], a_lo[0][0]);

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
__device__ void daxpy(double alpha, const double * x, double * y) {
  y[0] -= alpha * x[0]; y[1] -= alpha * x[1];
  y[2] -= alpha * x[2]; y[3] -= alpha * x[3];
}

// y(1:n) -= alpha * x(1:n)
__device__ void daxpy(int n, double alpha, const double * x, double * y) {
  y[0] -= alpha * x[0]; if (1 >= n) return; y[1] -= alpha * x[1]; if (2 >= n) return;
  y[2] -= alpha * x[2]; if (3 >= n) return; y[3] -= alpha * x[3];
}

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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 4) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ double a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ double b[mb][nb + 1];
    double x[8];

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
      double * X = B + i;

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
        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = A[0];

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
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

        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const double * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const double * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            daxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = A[0];

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];

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
      double * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];

        __syncthreads();

        // Start at the top of B and move down to X
        const double * _A = A;
        const double * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            daxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else
          a[threadIdx.x][threadIdx.y] = _A[0];

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] /= a[0][0]; daxpy(3, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(2, x[1], &a[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(1, x[2], &a[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] /= a[3][3];

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
      if (m > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (m > 1) { daxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (m > 2) { daxpy(m - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (m > 3) { daxpy(m - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}

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
    __shared__ double a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    double x[4];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      double * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        // Start at the left of B and move right to X
        const double * _A = A;
        const double * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = _A[0];

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            daxpy(_B[l * ldb], a[l], x);

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
          a[threadIdx.y][threadIdx.x] = _A[0];

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
        if (n > 1) { daxpy(3, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
        if (n > 2) { daxpy(2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
        if (n > 3) { daxpy(1, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}

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
      if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (n > 1) { daxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (n > 2) { daxpy(n - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (n > 3) { daxpy(n - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3]; }}}}

      // Write X
      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      double * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
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
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];

        __syncthreads();

        // Start one block beyond X and move to the right
        const double * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const double * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = _A[0];

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            daxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          daxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; daxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; daxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; daxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];

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
