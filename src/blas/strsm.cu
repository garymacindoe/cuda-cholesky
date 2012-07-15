#include "blas.h"

// y(1:8) -= alpha * x(1:8)
__device__ void saxpy(float alpha, const float * x, float * y) {
  y[0] -= alpha * x[0]; y[1] -= alpha * x[1];
  y[2] -= alpha * x[2]; y[3] -= alpha * x[3];
  y[4] -= alpha * x[4]; y[5] -= alpha * x[5];
  y[6] -= alpha * x[6]; y[7] -= alpha * x[7];
}

// y(1:n) -= alpha * x(1:n)
__device__ void saxpy(int n, float alpha, const float * x, float * y) {
  y[0] -= alpha * x[0]; if (1 >= n) return; y[1] -= alpha * x[1]; if (2 >= n) return;
  y[2] -= alpha * x[2]; if (3 >= n) return; y[3] -= alpha * x[3]; if (4 >= n) return;
  y[4] -= alpha * x[4]; if (5 >= n) return; y[5] -= alpha * x[5]; if (6 >= n) return;
  y[6] -= alpha * x[6]; if (7 >= n) return; y[7] -= alpha * x[7];
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
 * @param bx     blockDim.x.
 * @param by     blockDim.y.
 */
template <CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
          unsigned int mb, unsigned int nb, unsigned int bx, unsigned int by>
__global__ void strsm(int m, int n,
                      float alpha, const float * __restrict__ A, int lda,
                      float * __restrict__ B, int ldb) {

 if (side == CBlasLeft) {
   // For CBlasLeft each thread updates a column.  This means that B needs to be
   // read and written via shared memory to transpose it after reading it
   // efficiently from global memory.
//     typedef char _x[(mb == 8) ? 1 : -1];
//     typedef char _y[(nb == bx * by) ? 1 : -1];
//     typedef char _z[(bx == mb) ? 1 : -1];

    // Blocks of A and B is shared memory and X in registers
    __shared__ float a[mb][(transA == CBlasNoTrans) ? mb : mb + 1];
    __shared__ float b[mb][nb + 1];
    float x[8];

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
      float * X = B + i;

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
        x[4] = alpha * b[4][ti]; x[5] = alpha * b[5][ti]; x[6] = alpha * b[6][ti]; x[7] = alpha * b[7][ti];

        // Read the current block of A
        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = A[0];

        __syncthreads();

        // Update X from top to bottom
        switch (mm - 1) {
          case 7: if (diag == CBlasNonUnit) x[7] /= a[7][7]; saxpy(7, x[7], a[7], x);
          case 6: if (diag == CBlasNonUnit) x[6] /= a[6][6]; saxpy(6, x[6], a[6], x);
          case 5: if (diag == CBlasNonUnit) x[5] /= a[5][5]; saxpy(5, x[5], a[5], x);
          case 4: if (diag == CBlasNonUnit) x[4] /= a[4][4]; saxpy(4, x[4], a[4], x);
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; saxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; saxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; saxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
        }

        __syncthreads();

        // Write X out transposing it back via shared memory using b.
        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
        b[4][ti] = x[4]; b[5][ti] = x[5]; b[6][ti] = x[6]; b[7][ti] = x[7];

        __syncthreads();

        if (threadIdx.x < mm) {
          if (0 * by < n) { X[0 * by * ldb] = b[threadIdx.x][0 * by + threadIdx.y];
          if (1 * by < n) { X[1 * by * ldb] = b[threadIdx.x][1 * by + threadIdx.y];
          if (2 * by < n) { X[2 * by * ldb] = b[threadIdx.x][2 * by + threadIdx.y];
          if (3 * by < n) { X[3 * by * ldb] = b[threadIdx.x][3 * by + threadIdx.y];
          if (4 * by < n) { X[4 * by * ldb] = b[threadIdx.x][4 * by + threadIdx.y];
          if (5 * by < n) { X[5 * by * ldb] = b[threadIdx.x][5 * by + threadIdx.y];
          if (6 * by < n) { X[6 * by * ldb] = b[threadIdx.x][6 * by + threadIdx.y];
          if (7 * by < n) { X[7 * by * ldb] = b[threadIdx.x][7 * by + threadIdx.y]; }}}}}}}}
        }
      }

      // Move up to the next block
      A -= mb * lda + mb;
      X -= mb;
      i -= mb;

      while (i >= 0) {

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];
        x[4] = alpha * b[4][ti]; x[5] = alpha * b[5][ti]; x[6] = alpha * b[6][ti]; x[7] = alpha * b[7][ti];

        __syncthreads();

        const float * _A = A + ((transA == CBlasNoTrans) ? mb * lda : mb);
        const float * _B = X + mb;
        int k = m - i - mb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          if (k < mb) break;

          #pragma unroll
          for (int l = 0; l < mb; l++)
            saxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          saxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = A[0];
        else
          a[threadIdx.x][threadIdx.y] = A[0];

        __syncthreads();

        if (diag == CBlasNonUnit) x[7] /= a[7][7]; saxpy(7, x[7], a[7], x);
        if (diag == CBlasNonUnit) x[6] /= a[6][6]; saxpy(6, x[6], a[6], x);
        if (diag == CBlasNonUnit) x[5] /= a[5][5]; saxpy(5, x[5], a[5], x);
        if (diag == CBlasNonUnit) x[4] /= a[4][4]; saxpy(4, x[4], a[4], x);
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; saxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; saxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; saxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];

        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
        b[4][ti] = x[4]; b[5][ti] = x[5]; b[6][ti] = x[6]; b[7][ti] = x[7];

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
      float * X = B;
      int i = 0;

      while (m > 0) {
        #pragma unroll
        for (int j = 0; j < nb; j += by)
          b[threadIdx.x][j + threadIdx.y] = X[j * ldb];

        __syncthreads();

        x[0] = alpha * b[0][ti]; x[1] = alpha * b[1][ti]; x[2] = alpha * b[2][ti]; x[3] = alpha * b[3][ti];
        x[4] = alpha * b[4][ti]; x[5] = alpha * b[5][ti]; x[6] = alpha * b[6][ti]; x[7] = alpha * b[7][ti];

        __syncthreads();

        const float * _A = A;
        const float * _B = B;
        int k = i;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.y][threadIdx.x] = _A[0];
          else
            a[threadIdx.x][threadIdx.y] = _A[0];

          #pragma unroll
          for (int j = 0; j < nb; j += by)
            b[threadIdx.x][j + threadIdx.y] = _B[j * ldb];

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < mb; l++)
            saxpy(b[l][ti], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? mb * lda : mb;
          _B += mb;
          k -= mb;
        }

        for (int l = 0; l < k; l++)
          saxpy(b[l][ti], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.y][threadIdx.x] = _A[0];
        else
          a[threadIdx.x][threadIdx.y] = _A[0];

        __syncthreads();

        if (m < mb) break;

        if (diag == CBlasNonUnit) x[0] /= a[0][0]; saxpy(7, x[0], &a[0][1], &x[1]);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; saxpy(6, x[1], &a[1][2], &x[2]);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; saxpy(5, x[2], &a[2][3], &x[3]);
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; saxpy(4, x[3], &a[3][4], &x[4]);
        if (diag == CBlasNonUnit) x[4] /= a[4][4]; saxpy(3, x[4], &a[4][5], &x[5]);
        if (diag == CBlasNonUnit) x[5] /= a[5][5]; saxpy(2, x[5], &a[5][6], &x[6]);
        if (diag == CBlasNonUnit) x[6] /= a[6][6]; saxpy(1, x[6], &a[6][7], &x[7]);
        if (diag == CBlasNonUnit) x[7] /= a[7][7];

        b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
        b[4][ti] = x[4]; b[5][ti] = x[5]; b[6][ti] = x[6]; b[7][ti] = x[7];

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

      if (m > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (m > 1) { saxpy(m - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (m > 2) { saxpy(m - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (m > 3) { saxpy(m - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3];
      if (m > 4) { saxpy(m - 4, x[3], &a[3][4], &x[4]); if (diag == CBlasNonUnit) x[4] /= a[4][4];
      if (m > 5) { saxpy(m - 5, x[4], &a[4][5], &x[5]); if (diag == CBlasNonUnit) x[5] /= a[5][5];
      if (m > 6) { saxpy(m - 6, x[5], &a[5][6], &x[6]); if (diag == CBlasNonUnit) x[6] /= a[6][6];
      if (m > 7) { saxpy(m - 7, x[6], &a[6][7], &x[7]); if (diag == CBlasNonUnit) x[7] /= a[7][7]; }}}}}}}}

      __syncthreads();

      b[0][ti] = x[0]; b[1][ti] = x[1]; b[2][ti] = x[2]; b[3][ti] = x[3];
      b[4][ti] = x[4]; b[5][ti] = x[5]; b[6][ti] = x[6]; b[7][ti] = x[7];

      __syncthreads();

      if (threadIdx.x < m) {
        X[0] = b[threadIdx.x][by * 0 + threadIdx.y]; if (by * 1 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 1 + threadIdx.y]; if (by * 2 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 2 + threadIdx.y]; if (by * 3 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 3 + threadIdx.y]; if (by * 4 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 4 + threadIdx.y]; if (by * 5 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 5 + threadIdx.y]; if (by * 6 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 6 + threadIdx.y]; if (by * 7 >= n) return; X += by * ldb;
        X[0] = b[threadIdx.x][by * 7 + threadIdx.y];
      }
    }
  }
  else {
//     typedef char _x[(nb == 8) ? 1 : -1];
//     typedef char _y[(mb == bx * by) ? 1 : -1];
//     typedef char _z[(by == nb) ? 1 : -1];

    // Each thread computes a row of B 8 elements at a time
    __shared__ float a[nb][(transA == CBlasNoTrans) ? nb + 1 : nb];
    float x[8];

    const int ti = threadIdx.y * bx + threadIdx.x;

    // Compute the starting points in a and B for each thread
    A += threadIdx.y * lda + threadIdx.x;
    B += blockIdx.x * mb + ti;
    m -= blockIdx.x * mb;

    if ((uplo == CBlasUpper && transA == CBlasNoTrans) ||
        (uplo == CBlasLower && transA != CBlasNoTrans)) {
      float * X = B;
      int j = 0;

      while (n > 0) {
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];
        x[4] = alpha * X[4 * ldb]; x[5] = alpha * X[5 * ldb]; x[6] = alpha * X[6 * ldb]; x[7] = alpha * X[7 * ldb];

        const float * _A = A;
        const float * _B = B;
        int k = j;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = _A[0];

          __syncthreads();

          #pragma unroll
          for (int l = 0; l < nb; l++)
            saxpy(_B[l * ldb], a[l], x);

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

        if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
        if (n > 1) { saxpy(7, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
        if (n > 2) { saxpy(6, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
        if (n > 3) { saxpy(5, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3];
        if (n > 4) { saxpy(4, x[3], &a[3][4], &x[4]); if (diag == CBlasNonUnit) x[4] /= a[4][4];
        if (n > 5) { saxpy(3, x[4], &a[4][5], &x[5]); if (diag == CBlasNonUnit) x[5] /= a[5][5];
        if (n > 6) { saxpy(2, x[5], &a[5][6], &x[6]); if (diag == CBlasNonUnit) x[6] /= a[6][6];
        if (n > 7) { saxpy(1, x[6], &a[6][7], &x[7]); if (diag == CBlasNonUnit) x[7] /= a[7][7]; }}}}}}}}

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
          X[4 * ldb] = x[4]; X[5 * ldb] = x[5]; X[6 * ldb] = x[6]; X[7 * ldb] = x[7];
        }

        __syncthreads();

        A += (transA == CBlasNoTrans) ? nb * lda : nb;
        X += nb * ldb;
        n -= nb;
        j += nb;
      }

      if (n > 0) { if (diag == CBlasNonUnit) x[0] /= a[0][0];
      if (n > 1) { saxpy(n - 1, x[0], &a[0][1], &x[1]); if (diag == CBlasNonUnit) x[1] /= a[1][1];
      if (n > 2) { saxpy(n - 2, x[1], &a[1][2], &x[2]); if (diag == CBlasNonUnit) x[2] /= a[2][2];
      if (n > 3) { saxpy(n - 3, x[2], &a[2][3], &x[3]); if (diag == CBlasNonUnit) x[3] /= a[3][3];
      if (n > 4) { saxpy(n - 4, x[3], &a[3][4], &x[4]); if (diag == CBlasNonUnit) x[4] /= a[4][4];
      if (n > 5) { saxpy(n - 5, x[4], &a[4][5], &x[5]); if (diag == CBlasNonUnit) x[5] /= a[5][5];
      if (n > 6) { saxpy(n - 6, x[5], &a[5][6], &x[6]); if (diag == CBlasNonUnit) x[6] /= a[6][6];
      if (n > 7) { saxpy(n - 7, x[6], &a[6][7], &x[7]); if (diag == CBlasNonUnit) x[7] /= a[7][7]; }}}}}}}}

      if (ti < m) {
        X[0] = x[0]; if (1 >= n) return; X += ldb; X[0] = x[1]; if (2 >= n) return; X += ldb;
        X[0] = x[2]; if (3 >= n) return; X += ldb; X[0] = x[3]; if (4 >= n) return; X += ldb;
        X[0] = x[4]; if (5 >= n) return; X += ldb; X[0] = x[5]; if (6 >= n) return; X += ldb;
        X[0] = x[6]; if (7 >= n) return; X += ldb; X[0] = x[7];
      }
    }
    else {      /* (uplo == CBlasLower && transA == CBlasNoTrans) || (uplo == CBlasUpper && transA != CBlasNoTrans) */
      const int nn = n & (nb - 1);
      int j = n - nn;

      A += j * lda + j;
      float * X = B + j * ldb;

      if (nn > 0) {
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];
        x[4] = alpha * X[4 * ldb]; x[5] = alpha * X[5 * ldb]; x[6] = alpha * X[6 * ldb]; x[7] = alpha * X[7 * ldb];

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];

        __syncthreads();

        switch (nn - 1) {
          case 7: if (diag == CBlasNonUnit) x[7] /= a[7][7]; saxpy(7, x[7], a[7], x);
          case 6: if (diag == CBlasNonUnit) x[6] /= a[6][6]; saxpy(6, x[6], a[6], x);
          case 5: if (diag == CBlasNonUnit) x[5] /= a[5][5]; saxpy(5, x[5], a[5], x);
          case 4: if (diag == CBlasNonUnit) x[4] /= a[4][4]; saxpy(4, x[4], a[4], x);
          case 3: if (diag == CBlasNonUnit) x[3] /= a[3][3]; saxpy(3, x[3], a[3], x);
          case 2: if (diag == CBlasNonUnit) x[2] /= a[2][2]; saxpy(2, x[2], a[2], x);
          case 1: if (diag == CBlasNonUnit) x[1] /= a[1][1]; saxpy(1, x[1], a[1], x);
          case 0: if (diag == CBlasNonUnit) x[0] /= a[0][0];
        }

        if (ti < m) {
          X[0 * ldb] = x[0]; if (1 < nn) {
          X[1 * ldb] = x[1]; if (2 < nn) {
          X[2 * ldb] = x[2]; if (3 < nn) {
          X[3 * ldb] = x[3]; if (4 < nn) {
          X[4 * ldb] = x[4]; if (5 < nn) {
          X[5 * ldb] = x[5]; if (6 < nn) {
          X[6 * ldb] = x[6]; if (7 < nn) {
          X[7 * ldb] = x[7]; }}}}}}}
        }
      }

      A -= nb * lda + nb;
      X -= nb * ldb;
      j -= nb;

      while (j >= 0) {
        x[0] = alpha * X[0 * ldb]; x[1] = alpha * X[1 * ldb]; x[2] = alpha * X[2 * ldb]; x[3] = alpha * X[3 * ldb];
        x[4] = alpha * X[4 * ldb]; x[5] = alpha * X[5 * ldb]; x[6] = alpha * X[6 * ldb]; x[7] = alpha * X[7 * ldb];

        __syncthreads();

        const float * _A = A + ((transA == CBlasNoTrans) ? nb : nb * lda);
        const float * _B = X + nb * ldb;
        int k = n - j - nb;
        while (k > 0) {

          if (transA == CBlasNoTrans)
            a[threadIdx.x][threadIdx.y] = _A[0];
          else
            a[threadIdx.y][threadIdx.x] = _A[0];

          __syncthreads();

          if (k < nb) break;

          #pragma unroll
          for (int l = 0; l < nb; l++)
            saxpy(_B[l * ldb], a[l], x);

          __syncthreads();

          _A += (transA == CBlasNoTrans) ? nb : nb * lda;
          _B += nb * ldb;
          k -= nb;
        }

        for (int l = 0; l < k; l++)
          saxpy(_B[l * ldb], a[l], x);

        __syncthreads();

        if (transA == CBlasNoTrans)
          a[threadIdx.x][threadIdx.y] = A[0];
        else
          a[threadIdx.y][threadIdx.x] = A[0];

        __syncthreads();

        if (diag == CBlasNonUnit) x[7] /= a[7][7]; saxpy(7, x[7], a[7], x);
        if (diag == CBlasNonUnit) x[6] /= a[6][6]; saxpy(6, x[6], a[6], x);
        if (diag == CBlasNonUnit) x[5] /= a[5][5]; saxpy(5, x[5], a[5], x);
        if (diag == CBlasNonUnit) x[4] /= a[4][4]; saxpy(4, x[4], a[4], x);
        if (diag == CBlasNonUnit) x[3] /= a[3][3]; saxpy(3, x[3], a[3], x);
        if (diag == CBlasNonUnit) x[2] /= a[2][2]; saxpy(2, x[2], a[2], x);
        if (diag == CBlasNonUnit) x[1] /= a[1][1]; saxpy(1, x[1], a[1], x);
        if (diag == CBlasNonUnit) x[0] /= a[0][0];

        if (ti < m) {
          X[0 * ldb] = x[0]; X[1 * ldb] = x[1]; X[2 * ldb] = x[2]; X[3 * ldb] = x[3];
          X[4 * ldb] = x[4]; X[5 * ldb] = x[5]; X[6 * ldb] = x[6]; X[7 * ldb] = x[7];
        }

        A -= nb * lda + nb;
        X -= nb * ldb;
        j -= nb;
      }
    }
  }
}

template void strsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasNonUnit,  8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasNoTrans, CBlasUnit,     8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasNonUnit,  8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasUpper, CBlasTrans,   CBlasUnit,     8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasNonUnit,  8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasNoTrans, CBlasUnit,     8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasNonUnit,  8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasLeft,  CBlasLower, CBlasTrans,   CBlasUnit,     8, 64, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, 64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit,    64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasNonUnit, 64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasUpper, CBlasTrans,   CBlasUnit,    64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, 64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit,    64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasNonUnit, 64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
template void strsm<CBlasRight, CBlasLower, CBlasTrans,   CBlasUnit,    64,  8, 8, 8>(int, int, float, const float * __restrict__, int, float * __restrict__, int);
