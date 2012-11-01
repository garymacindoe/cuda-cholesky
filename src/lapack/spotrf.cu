// nvcc -I../../include -O2 -arch=compute_13 -code=sm_13 -use_fast_math -Xptxas=-v -maxrregcount=32 -cubin spotrf.cu
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
__global__ void spotf2(int n, float * A, int lda, int * info) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sinfo;

  // thread 0 is the only thread to write to info in shared or global memory
  if (threadIdx.x == 0)
    *info = sinfo = 0;  // initialise info to zero and cache

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 64 threads are needed but this would
   * use all 16384 bytes (64 * 64 * sizeof(float)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 8320 bytes((64 * (64 + 1)) / 2 * sizeof(float))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ float a[(bx * (bx + 1)) / 2];

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
      float temp;
      if (threadIdx.x >= j) {
        // SGEMV/SSYRK
        temp = a[upper(j, threadIdx.x)];
        for (int k = 0; k < j; k++)
          temp -= a[upper(k, j)] * a[upper(k, threadIdx.x)];

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (temp <= 0.0f || isnan(temp)) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[upper(j, threadIdx.x)] = temp;
          }
          else
            a[upper(j, threadIdx.x)] = sqrtf(temp);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // SSCAL
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
      float temp;
      if (threadIdx.x >= j) {
        // SGEMV/SSYRK
        temp = a[lower<bx>(threadIdx.x, j)];
        for (int k = 0; k < j; k++)
          temp -= a[lower<bx>(j, k)] * a[lower<bx>(threadIdx.x, k)];

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (temp <= 0.0f || isnan(temp)) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[lower<bx>(threadIdx.x, j)] = temp;
          }
          else
            a[lower<bx>(threadIdx.x, j)] = sqrtf(temp);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // SSCAL
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

template void spotf2<CBlasUpper, 64>(int, float *, int, int *);
template void spotf2<CBlasLower, 64>(int, float *, int, int *);
