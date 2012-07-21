#include "blas.h"
#include <cuComplex.h>

#if __CUDA_ARCH__ < 200 && !defined(__BANK_CONFLICT__)

// y(1:2) -= alpha * x(1:2)
__device__ void zaxpy(cuDoubleComplex alpha, const int * x_real_hi, const int * x_real_lo,
                      const int * x_imag_hi, const int * x_imag_lo, cuDoubleComplex * y) {
  y[0] = cuCsub(y[0], cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                                         __hiloint2double(x_imag_hi[0], x_imag_lo[0]))));
  y[1] = cuCsub(y[1], cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                                         __hiloint2double(x_imag_hi[1], x_imag_lo[1]))));
}

// y(1:n) -= alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex alpha, const int * x_real_hi, const int * x_real_lo,
                      const int * x_imag_hi, const int * x_imag_lo, cuDoubleComplex * y) {
  y[0] = cuCsub(y[0], cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[0], x_real_lo[0]),
                                                         __hiloint2double(x_imag_hi[0], x_imag_lo[0]))));
  if (1 >= n) return;
  y[1] = cuCsub(y[1], cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(x_real_hi[1], x_real_lo[1]),
                                                         __hiloint2double(x_imag_hi[1], x_imag_lo[1]))));
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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 2) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ int a_real_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_real_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_imag_hi[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int a_imag_lo[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ int b_real_hi[mb][nb + 1];
    __shared__ int b_real_lo[mb][nb + 1];
    __shared__ int b_imag_hi[mb][nb + 1];
    __shared__ int b_imag_lo[mb][nb + 1];
    cuDoubleComplex x[2];

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
      cuDoubleComplex * X = B + i;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (mm > 0) {
        // Read the block of B we are updating and transpose into shared
        // memory using b.
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(X[j * ldb]));
          b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(X[j * ldb]));
          b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(X[j * ldb]));
          b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(X[j * ldb]));
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[0][ti], b_real_lo[0][ti]),
                                                  __hiloint2double(b_imag_hi[0][ti], b_imag_lo[0][ti])));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[1][ti], b_real_lo[1][ti]),
                                                  __hiloint2double(b_imag_hi[1][ti], b_imag_lo[1][ti])));

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[0]));
        }
        else {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
        }

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 1:
            if (diag == CBlasNonUnit)
              x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                       __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
            zaxpy(1, x[1], a_real_hi[1], a_real_lo[1], a_imag_hi[1], a_imag_lo[1], x);
          case 0:
            if (diag == CBlasNonUnit)
              x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                       __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b_real_hi[0][ti] = __double2hiint(cuCreal(x[0])); b_real_lo[0][ti] = __double2loint(cuCreal(x[0]));
        b_imag_hi[0][ti] = __double2hiint(cuCimag(x[0])); b_imag_lo[0][ti] = __double2loint(cuCimag(x[0]));
        b_real_hi[1][ti] = __double2hiint(cuCreal(x[1])); b_real_lo[1][ti] = __double2loint(cuCreal(x[1]));
        b_imag_hi[1][ti] = __double2hiint(cuCimag(x[1])); b_imag_lo[1][ti] = __double2loint(cuCimag(x[1]));

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) {
            X[0 * by * ldb] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][0 * by + threadIdx.y], b_real_lo[threadIdx.x][0 * by + threadIdx.y]),
                                                   __hiloint2double(b_imag_hi[threadIdx.x][0 * by + threadIdx.y], b_imag_lo[threadIdx.x][0 * by + threadIdx.y]));
            if (1 * by < n)
              X[1 * by * ldb] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][1 * by + threadIdx.y], b_real_lo[threadIdx.x][1 * by + threadIdx.y]),
                                                     __hiloint2double(b_imag_hi[threadIdx.x][1 * by + threadIdx.y], b_imag_lo[threadIdx.x][1 * by + threadIdx.y]));
          }
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
          b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(X[j * ldb]));
          b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(X[j * ldb]));
          b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(X[j * ldb]));
          b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(X[j * ldb]));
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[0][ti], b_real_lo[0][ti]),
                                                  __hiloint2double(b_imag_hi[0][ti], b_imag_lo[0][ti])));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[1][ti], b_real_lo[1][ti]),
                                                  __hiloint2double(b_imag_hi[1][ti], b_imag_lo[1][ti])));

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuDoubleComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(_A[0]));
            a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCimag(_A[0]));
          }
          else {
            a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
            a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(_B[j * ldb]));
            b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(_B[j * ldb]));
            b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(_B[j * ldb]));
            b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(_B[j * ldb]));
          }

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(make_cuDoubleComplex(__hiloint2double(b_real_hi[l][ti], b_real_lo[l][ti]),
                                       __hiloint2double(b_imag_hi[l][ti], b_imag_lo[l][ti])),
                  a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(__hiloint2double(b_real_hi[l][ti], b_real_lo[l][ti]),
                                     __hiloint2double(b_imag_hi[l][ti], b_imag_lo[l][ti])),
                a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCimag(A[0]));
        }
        else {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
        }

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit)
          x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                   __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
        zaxpy(1, x[1], a_real_hi[1], a_real_lo[1], a_imag_hi[1], a_imag_lo[1], x);
        if (diag == CBlasNonUnit)
          x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                   __hiloint2double(a_imag_hi[0][0], a_imag_hi[0][0])));

        // Write X out transposing it back via shared memory using b
        b_real_hi[0][ti] = __double2hiint(cuCreal(x[0])); b_real_lo[0][ti] = __double2loint(cuCreal(x[0]));
        b_imag_hi[0][ti] = __double2hiint(cuCimag(x[0])); b_imag_lo[0][ti] = __double2loint(cuCimag(x[0]));
        b_real_hi[1][ti] = __double2hiint(cuCreal(x[1])); b_real_lo[1][ti] = __double2loint(cuCreal(x[1]));
        b_imag_hi[1][ti] = __double2hiint(cuCimag(x[1])); b_imag_lo[1][ti] = __double2loint(cuCimag(x[1]));

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][j + threadIdx.y], b_real_lo[threadIdx.x][j + threadIdx.y]),
                                            __hiloint2double(b_imag_hi[threadIdx.x][j + threadIdx.y], b_imag_lo[threadIdx.x][j + threadIdx.y]));

        // Move up to the next blocks of A and B (through X)
        A -= mb * lda + mb;
        X -= mb;
        i -= mb;
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case we start at the top of B and work downwards
      cuDoubleComplex * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by) {
          b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(X[j * ldb]));
          b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(X[j * ldb]));
          b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(X[j * ldb]));
          b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(X[j * ldb]));
        }

        __syncthreads();

        // Place it into X as alpha * B
        x[0] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[0][ti], b_real_lo[0][ti]),
                                                  __hiloint2double(b_imag_hi[0][ti], b_imag_lo[0][ti])));
        x[1] = cuCmul(alpha, make_cuDoubleComplex(__hiloint2double(b_real_hi[1][ti], b_real_lo[1][ti]),
                                                  __hiloint2double(b_imag_hi[1][ti], b_imag_lo[1][ti])));

        __syncthreads();

        // Start at the top of B and move down to X
        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans) {
            a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(_A[0]));
            a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCimag(_A[0]));
          }
          else {
            a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
            a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          }

          #pragma unroll
          for (int j = 0; j < nb; j += by) {
            b_real_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCreal(_B[j * ldb]));
            b_real_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCreal(_B[j * ldb]));
            b_imag_hi[threadIdx.x][j + threadIdx.y] = __double2hiint(cuCimag(_B[j * ldb]));
            b_imag_lo[threadIdx.x][j + threadIdx.y] = __double2loint(cuCimag(_B[j * ldb]));
          }

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
          zaxpy(make_cuDoubleComplex(__hiloint2double(b_real_hi[l][ti], b_real_lo[l][ti]),
                                     __hiloint2double(b_imag_hi[l][ti], b_imag_lo[l][ti])),
                a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(make_cuDoubleComplex(__hiloint2double(b_real_hi[l][ti], b_real_lo[l][ti]),
                                     __hiloint2double(b_imag_hi[l][ti], b_imag_lo[l][ti])),
                a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCimag(_A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCimag(_A[0]));
        }
        else {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
        }

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit)
          x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                   __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
        zaxpy(1, x[0], &a_real_hi[0][1], &a_real_lo[0][1], &a_imag_hi[0][1], &a_imag_lo[0][1], &x[1]);
        if (diag == CBlasNonUnit)
          x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                   __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));

        // Write X out transposing it back via shared memory using b
        b_real_hi[0][ti] = __double2hiint(cuCreal(x[0])); b_real_lo[0][ti] = __double2loint(cuCreal(x[0]));
        b_imag_hi[0][ti] = __double2hiint(cuCimag(x[0])); b_imag_lo[0][ti] = __double2loint(cuCimag(x[0]));
        b_real_hi[1][ti] = __double2hiint(cuCreal(x[1])); b_real_lo[1][ti] = __double2loint(cuCreal(x[1]));
        b_imag_hi[1][ti] = __double2hiint(cuCimag(x[1])); b_imag_lo[1][ti] = __double2loint(cuCimag(x[1]));

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          X[j * ldb] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][j + threadIdx.y], b_real_lo[threadIdx.x][j + threadIdx.y]),
                                            __hiloint2double(b_imag_hi[threadIdx.x][j + threadIdx.y], b_imag_lo[threadIdx.x][j + threadIdx.y]));

        __syncthreads();

        // Move up to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? mb : mb * lda;
        X += mb;
        m -= mb;
        i += mb;
      }

      // Handle the trailing elements last, if any.
      if (m > 0) {
        if (diag == CBlasNonUnit)
          x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                   __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
        if (m > 1) {
          zaxpy(m - 1, x[0], &a_real_hi[0][1], &a_real_lo[0][1], &a_imag_hi[0][1], &a_imag_lo[0][1], &x[1]);
          if (diag == CBlasNonUnit)
            x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                     __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
        }
      }

      __syncthreads();

      // Write X out transposing it back via shared memory using b
      b_real_hi[0][ti] = __double2hiint(cuCreal(x[0])); b_real_lo[0][ti] = __double2loint(cuCreal(x[0]));
      b_imag_hi[0][ti] = __double2hiint(cuCimag(x[0])); b_imag_lo[0][ti] = __double2loint(cuCimag(x[0]));
      b_real_hi[1][ti] = __double2hiint(cuCreal(x[1])); b_real_lo[1][ti] = __double2loint(cuCreal(x[1]));
      b_imag_hi[1][ti] = __double2hiint(cuCimag(x[1])); b_imag_lo[1][ti] = __double2loint(cuCimag(x[1]));

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][0 * by + threadIdx.y], b_real_lo[threadIdx.x][0 * by + threadIdx.y]),
                                    __hiloint2double(b_imag_hi[threadIdx.x][0 * by + threadIdx.y], b_imag_lo[threadIdx.x][0 * by + threadIdx.y]));
        if (by * 1 >= n) return; X += by * ldb;
        X[0] = make_cuDoubleComplex(__hiloint2double(b_real_hi[threadIdx.x][1 * by + threadIdx.y], b_real_lo[threadIdx.x][1 * by + threadIdx.y]),
                                    __hiloint2double(b_imag_hi[threadIdx.x][1 * by + threadIdx.y], b_imag_lo[threadIdx.x][1 * by + threadIdx.y]));
      }
    }
  }
  else {
   // For CBlasRight each thread updates a row.  This means that B can be read
   // efficiently straight from global memory.
//     typedef char _x[(nb == 2) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 2 elements at a time
    __shared__ int a_real_hi[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ int a_real_lo[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ int a_imag_hi[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    __shared__ int a_imag_lo[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuDoubleComplex x[2];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      cuDoubleComplex * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        // Start at the left of B and move right to X
        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans) {
            a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCimag(_A[0]));
            a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCimag(_A[0]));
          }
          else {
            a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
            a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          }

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

          __syncthreads();

          //  Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCimag(_A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCimag(_A[0]));
        }
        else {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
        }

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (n > 0) {
          if (diag == CBlasNonUnit)
            x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                     __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
          if (n > 1) {
            zaxpy(1, x[0], &a_real_hi[0][1], &a_real_lo[0][1], &a_imag_hi[0][1], &a_imag_lo[0][1], &x[1]);
            if (diag == CBlasNonUnit)
              x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                       __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
          }
        }

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
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
        if (diag == CBlasNonUnit)
          x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                   __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
        if (n > 1) {
          zaxpy(n - 1, x[0], &a_real_hi[0][1], &a_real_lo[0][1], &a_imag_hi[0][1], &a_imag_lo[0][1], &x[1]);
          if (diag == CBlasNonUnit)
            x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                     __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
        }
      }

      // Write X
      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuDoubleComplex * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        // Read the current block of A
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCimag(A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCimag(A[0]));
        }
        else {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
        }

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 1:
            if (diag == CBlasNonUnit)
              x[1] = cuCdiv(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                       __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
            zaxpy(1, x[1], a_real_hi[1], a_real_lo[1], a_imag_hi[1], a_imag_lo[1], x);
          case 0:
            if (diag == CBlasNonUnit)
              x[0] = cuCdiv(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                       __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));
        }

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0];
          if (1 < nn)
            X[1 * ldb] = x[1];
        }
      }

      // Move left to the next block
      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        // Read the current block of X and multiply by alpha
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        __syncthreads();

        // Start one block beyond X and move to the right
        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuDoubleComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans) {
            a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCimag(_A[0]));
            a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCimag(_A[0]));
          }
          else {
            a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(_A[0]));
            a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(_A[0]));
            a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
            a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(_A[0]) : cuCimag(_A[0]));
          }

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(_B[l * ldb], a_real_hi[l], a_real_lo[l], a_imag_hi[l], a_imag_lo[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans) {
          a_real_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.x][threadIdx.y] = __double2hiint(cuCimag(A[0]));
          a_imag_lo[threadIdx.x][threadIdx.y] = __double2loint(cuCimag(A[0]));
        }
        else {
          a_real_hi[threadIdx.y][threadIdx.x] = __double2hiint(cuCreal(A[0]));
          a_real_lo[threadIdx.y][threadIdx.x] = __double2loint(cuCreal(A[0]));
          a_imag_hi[threadIdx.y][threadIdx.x] = __double2hiint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
          a_imag_lo[threadIdx.y][threadIdx.x] = __double2loint((transA == CBlasConjTrans) ? -cuCimag(A[0]) : cuCimag(A[0]));
        }

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit)
          x[1] = cuCmul(x[1], make_cuDoubleComplex(__hiloint2double(a_real_hi[1][1], a_real_lo[1][1]),
                                                   __hiloint2double(a_imag_hi[1][1], a_imag_lo[1][1])));
        zaxpy(1, x[1], a_real_hi[1], a_real_lo[1], a_imag_hi[1], a_imag_lo[1], x);
        if (diag == CBlasNonUnit)
          x[0] = cuCmul(x[0], make_cuDoubleComplex(__hiloint2double(a_real_hi[0][0], a_real_lo[0][0]),
                                                   __hiloint2double(a_imag_hi[0][0], a_imag_lo[0][0])));

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
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

// y(1:2) -= alpha * x(1:2)
__device__ void zaxpy(cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCsub(y[0], cuCmul(alpha, x[0]));
  y[1] = cuCsub(y[1], cuCmul(alpha, x[1]));
}

// y(1:n) -= alpha * x(1:n)
__device__ void zaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex * x, cuDoubleComplex * y) {
  y[0] = cuCsub(y[0], cuCmul(alpha, x[0])); if (1 >= n) return;
  y[1] = cuCsub(y[1], cuCmul(alpha, x[1]));
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
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 2) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ cuDoubleComplex a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ cuDoubleComplex b[mb][nb + 1];
    cuDoubleComplex x[2];

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
      cuDoubleComplex * X = B + i;

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
        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConj(A[0]) : A[0];

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b[0][ti] = x[0]; b[1][ti] = x[1];

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y]; }}
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

        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        __syncthreads();

        // Start at the block one beyond X and move to the bottom
        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const cuDoubleComplex * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConj(A[0]) : A[0];

        __syncthreads();

        // Update X unrolled (reverse loop)
        if (diag == CBlasNonUnit) x[1] = cuCmul(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCmul(x[0], a[0][0]);

        // Write X out transposing it back via shared memory using b
        b[0][ti] = x[0]; b[1][ti] = x[1];

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
      cuDoubleComplex * X = B;
      int i = 0;

      while (m > 0) {
        // Read the current block of X
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = cuCmul(alpha, b[0][ti]); x[1] = cuCmul(alpha, b[1][ti]);

        __syncthreads();

        // Start at the top of B and move down to X
        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = i;
        while (k > 0) {

          // Read A and B into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < mb; l++)
            zaxpy(b[l][ti], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(b[l][ti], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else
          a[threadIdx.x][threadIdx.y] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

        __syncthreads();

        if (m < mb) break;

        // Update X unrolled (forward loop)
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]); zaxpy(3, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]);

        // Write X out transposing it back via shared memory using b
        b[0][ti] = x[0]; b[1][ti] = x[1];

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
      if (m > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
      if (m > 1) { zaxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

      __syncthreads();

      // Write X out transposing it back via shared memory using b
      b[0][ti] = x[0]; b[1][ti] = x[1];

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = b[threadIdx.x][by * 0 + threadIdx.y]; if (by * 1 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 1 + threadIdx.y];
      }
    }
  }
  else {
   // For CBlasRight each thread updates a row.  This means that B can be read
   // efficiently straight from global memory.
//     typedef char _x[(nb == 2) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 4 elements at a time
    __shared__ cuDoubleComplex a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    cuDoubleComplex x[2];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in A and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    // There are 2 common cases for each of CBlasLeft and CBlasRight
    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      // For this case start on the left and work right
      cuDoubleComplex * X = B;
      int j = 0;

      while (n > 0) {
        // Read the current block of X
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        // Start at the left of B and move right to X
        const cuDoubleComplex * _A = A;
        const cuDoubleComplex * _B = B;
        int k = j;
        while (k > 0) {

          // Read A into shared memory
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

          __syncthreads();

          // Update X reading B straight from global memory
          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a[l], x);

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
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

        __syncthreads();

        if (n < nb) break;

        // Update X unrolled (forward loop)
        if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        if (n > 1) { zaxpy(3, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
        }

        __syncthreads();

        // Move right to the next blocks of A and B (through X)
        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      // Update X unrolled (forward loop)
      if (n > 0) { if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
      if (n > 1) { zaxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); }}

      // Write X
      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      // For this case start on the right and work left
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      cuDoubleComplex * X = B + j * ldb;

      // Handle the trailing elements first, if any.  This only requires reading
      // the block of B that we are also updating (X == _B).
      if (nn > 0) {
        // Read the block of B we are updating
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConj(A[0]) : A[0];

        __syncthreads();

        // Update X from right to left
        switch (nn - 1) {
          case 1: if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);
        }

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn)
          X[1 * ldb] = x[1];
        }
      }

      // Move left to the next block
      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        // Read the current block of X and multiply by alpha
        x[0] = cuCmul(alpha, X[0 * ldb]); x[1] = cuCmul(alpha, X[1 * ldb]);

        __syncthreads();

        // Start one block beyond X and move to the right
        const cuDoubleComplex * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const cuDoubleComplex * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          // Read the current block of A
          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConj(_A[0]) : _A[0];

          __syncthreads();

          if (k < nb) break;

          // Update X in registers
          #pragma unroll
          for (int l = 0; l < nb; l++)
            zaxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          // Move to the next blocks of A and B
          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        // Process odd elements of A and B
        for (int l = 0; l < k; l++)
          zaxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        // Read the block of A that matches the block of B which is in registers
        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = (transA == CBlasConjTrans) ? cuConj(A[0]) : A[0];

        __syncthreads();

        // Update X from right to left
        if (diag == CBlasNonUnit) x[1] = cuCdiv(x[1], a[1][1]); zaxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] = cuCdiv(x[0], a[0][0]);

        // Write X
        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1];
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

template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit, 2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,    2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit, 2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,    2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit, 2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,    2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit, 2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,    2, 8, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
template void ztrsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    8, 2, 2, 2>(int, int, cuDoubleComplex, const cuDoubleComplex * __restrict__, int, cuDoubleComplex * __restrict__, int);
