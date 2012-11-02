#include "blas.h"

/*
 * Indexing function for upper triangular packed storage mode.  Only works when
 * i <= j otherwise generates an out-of-bounds access in shared memory and CUDA
 * will segfault.
 */
__device__ int upper(int i, int j) {
  return ((j * (j + 1)) / 2) + i;
}

/*
 * Indexing function for lower triangular packed storage mode.  Only works when
 * i >= j otherwise generates an out-of-bounds access in shared memory and CUDA
 * will segfault.
 */
template <unsigned int bx>
__device__ int lower(int i, int j) {
  return ((2 * bx - j - 1) * j) / 2 + i;
}

template <CBlasUplo uplo, unsigned int bx>
__global__ void dpotf2(int n, double * A, int lda, int * info) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sinfo;

  // thread 0 is the only thread to write to info in shared or global memory
  if (threadIdx.x == 0)
    *info = sinfo = 0;  // initialise info to zero and cache

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 32 threads are needed but this would
   * use 8192 bytes (32 * 32 * sizeof(double)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 4224 bytes((32 * (32 + 1)) / 2 * sizeof(double))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ double a[(bx * (bx + 1)) / 2];

  if (uplo == CBlasUpper) {
    // Read upper triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (threadIdx.x <= j)
        a[upper(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the cholesky decomposition
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also neatly avoids
    // bank conflicts.
    for (int j = 0; j < n; j++) {
      double temp;
      if (threadIdx.x >= j) {
        // DGEMV/DSYRK
        temp = a[upper(j, threadIdx.x)];
        for (int k = 0; k < j; k++)
          temp -= a[upper(k, j)] * a[upper(k, threadIdx.x)];

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (temp <= 0.0 || isnan(temp)) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[upper(j, threadIdx.x)] = temp;
          }
          else
            a[upper(j, threadIdx.x)] = sqrt(temp);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // DSCAL
      if (threadIdx.x > j)
        a[upper(j, threadIdx.x)] = temp / a[upper(j, j)];

      __syncthreads();
    }

    // Write the upper triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x <= j)
        A[j * lda + threadIdx.x] = a[upper(threadIdx.x, j)];
    }
  }
  else {
    // Read lower triangle of A into shared memory
    #pragma unroll
    for (int j = 0; j < bx; j++) {
      if (threadIdx.x >= j)
        a[lower<bx>(threadIdx.x, j)] = A[j * lda + threadIdx.x];
    }

    __syncthreads();

    // Perform the cholesky decomposition
    // Accesses do not have to be coalesced or aligned as they would if A were
    // in global memory.  Using triangular packed storage also neatly avoids
    // bank conflicts.
    for (int j = 0; j < n; j++) {
      double temp;
      if (threadIdx.x >= j) {
        // DGEMV/DSYRK
        temp = a[lower<bx>(threadIdx.x, j)];
        for (int k = 0; k < j; k++)
          temp -= a[lower<bx>(j, k)] * a[lower<bx>(threadIdx.x, k)];

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (temp <= 0.0 || isnan(temp)) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[lower<bx>(threadIdx.x, j)] = temp;
          }
          else
            a[lower<bx>(threadIdx.x, j)] = sqrt(temp);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // DSCAL
      if (threadIdx.x > j)
        a[lower<bx>(threadIdx.x, j)] = temp / a[lower<bx>(j, j)];

      __syncthreads();
    }

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

template void dpotf2<CBlasUpper, 32>(int, double *, int, int *);
template void dpotf2<CBlasLower, 32>(int, double *, int, int *);
