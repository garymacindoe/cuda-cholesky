#include "blas.h"
#include <cuComplex.h>

/*
 * cuComplex.h provides cuCfmaf but not cuCfsmf.
 *
 * cuCfmaf: x * y + d
 * cuCfmsf: x * y - d
 * cuCfsmf: d - x * y
 */
__device__ cuComplex cuCfsmf(cuComplex x, cuComplex y, cuComplex d) {
    float real_res;
    float imag_res;

    real_res = -(cuCrealf(x) * cuCrealf(y)) + cuCrealf(d);
    imag_res = -(cuCrealf(x) * cuCimagf(y)) + cuCimagf(d);

    real_res =  (cuCimagf(x) * cuCimagf(y)) + real_res;
    imag_res = -(cuCimagf(x) * cuCrealf(y)) + imag_res;

    return make_cuComplex(real_res, imag_res);
}

/*
 * Divide complex by scalar.
 */
__device__ cuComplex cusDivf(cuComplex x, float a) {
  return make_cuComplex(cuCrealf(x) / a, cuCimagf(x) / a);
}

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
__global__ void cpotf2(int n, cuComplex * A, int lda, int * info) {
  // info parameter cached in shared memory for fast access by all threads in the block
  __shared__ int sinfo;

  // thread 0 is the only thread to write to info in shared or global memory
  if (threadIdx.x == 0)
    *info = sinfo = 0;  // initialise info to zero and cache

  /*
   * For efficient data reuse A needs to be cached in shared memory.  In order
   * to get maximum instruction throughput 32 threads are needed but this would
   * use 8192 bytes (32 * 32 * sizeof(cuComplex)) of shared memory to store A.
   * Triangular packed storage mode is therefore used to store only the
   * triangle of A being updated using 4224 bytes((32 * (32 + 1)) / 2 * sizeof(cuComplex))
   * of shared memory.
   * Since this is only ever going to be run using one thread block shared
   * memory and register use can be higher than when trying to fit multiple
   * thread blocks onto each multiprocessor.
   */
  __shared__ cuComplex a[(bx * (bx + 1)) / 2];

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
      cuComplex temp;
      if (threadIdx.x >= j) {
        // CGEMV/CHERK
        temp = a[upper(j, threadIdx.x)];
        for (int k = 0; k < j; k++)
          temp = cuCsubf(temp, cuCmulf(a[upper(k, j)], cuConjf(a[upper(k, threadIdx.x)])));//cuCfsmf(a[upper(k, j)], cuConjf(a[upper(k, threadIdx.x)]), temp);

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (cuCrealf(temp) <= 0.0f || isnan(cuCrealf(temp))) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[upper(j, threadIdx.x)] = temp;
          }
          else
            a[upper(j, threadIdx.x)] = make_cuComplex(sqrtf(cuCrealf(temp)), 0.0f);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // CSSCAL
      if (threadIdx.x > j)
        a[upper(j, threadIdx.x)] = make_cuComplex(cuCrealf(temp) / cuCrealf(a[upper(j, j)]), cuCimagf(temp) / cuCrealf(a[upper(j, j)]));//cusDivf(temp, cuCrealf(a[upper(j, j)]));

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
      cuComplex temp;
      if (threadIdx.x >= j) {
        // CGEMV/CHERK
        temp = a[lower<bx>(threadIdx.x, j)];
        for (int k = 0; k < j; k++)
          temp = cuCfsmf(a[lower<bx>(j, k)], cuConjf(a[lower<bx>(threadIdx.x, k)]), temp);

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (cuCrealf(temp) <= 0.0f || isnan(cuCrealf(temp))) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[lower<bx>(threadIdx.x, j)] = make_cuComplex(cuCrealf(temp), 0.0f);
          }
          else
            a[lower<bx>(threadIdx.x, j)] = make_cuComplex(sqrt(cuCrealf(temp)), 0.0f);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // CSSCAL
      if (threadIdx.x > j)
        a[lower<bx>(threadIdx.x, j)] = cusDivf(temp, cuCrealf(a[lower<bx>(j, j)]));

      __syncthreads();
    }

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

template void cpotf2<CBlasUpper, 32>(int, cuComplex *, int, int *);
template void cpotf2<CBlasLower, 32>(int, cuComplex *, int, int *);
