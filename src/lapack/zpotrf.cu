#include "blas.h"
#include <cuComplex.h>

/*
 * cuComplex.h provides cuCfma but not cuCfsm.
 *
 * cuCfma: x * y + d
 * cuCfms: x * y - d
 * cuCfsm: d - x * y
 */
__device__ cuDoubleComplex cuCfsm(cuDoubleComplex x, cuDoubleComplex y, cuDoubleComplex d) {
    double real_res;
    double imag_res;

    real_res = -(cuCreal(x) * cuCreal(y)) + cuCreal(d);
    imag_res = -(cuCreal(x) * cuCimag(y)) + cuCimag(d);

    real_res =  (cuCimag(x) * cuCimag(y)) + real_res;
    imag_res = -(cuCimag(x) * cuCreal(y)) + imag_res;

    return make_cuDoubleComplex(real_res, imag_res);
}

/*
 * Divide complex by scalar.
 */
__device__ cuDoubleComplex cuDiva(cuDoubleComplex x, double a) {
  return make_cuDoubleComplex(cuCreal(x) / a, cuCimag(x) / a);
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
__global__ void zpotf2(int n, cuDoubleComplex * A, int lda, int * info) {
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
  __shared__ cuDoubleComplex a[(bx * (bx + 1)) / 2];

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
      cuDoubleComplex temp;
      if (threadIdx.x >= j) {
        // ZGEMV/ZHERK
        temp = a[upper(j, threadIdx.x)];
        for (int k = 0; k < j; k++)
          temp = cuCfsm(a[upper(k, j)], cuConj(a[upper(k, threadIdx.x)]), temp);

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (cuCreal(temp) <= 0.0 || isnan(cuCreal(temp))) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[upper(j, threadIdx.x)] = temp;
          }
          else
            a[upper(j, threadIdx.x)] = make_cuDoubleComplex(sqrt(cuCreal(temp)), 0.0);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // ZSSCAL
      if (threadIdx.x > j)
        a[upper(j, threadIdx.x)] = cuDiva(temp, cuCreal(a[upper(j, j)]));

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
      cuDoubleComplex temp;
      if (threadIdx.x >= j) {
        // ZGEMV/ZHERK
        temp = a[lower<bx>(threadIdx.x, j)];
        for (int k = 0; k < j; k++)
          temp = cuCfsm(a[lower<bx>(j, k)], cuConj(a[lower<bx>(threadIdx.x, k)]), temp);

        // Thread zero calculates the diagonal element
        if (threadIdx.x == j) {
          if (cuCreal(temp) <= 0.0 || isnan(cuCreal(temp))) {
            *info = sinfo = j + 1;      // update info in shared and global memory
            a[lower<bx>(threadIdx.x, j)] = make_cuDoubleComplex(cuCreal(temp), 0.0);
          }
          else
            a[lower<bx>(threadIdx.x, j)] = make_cuDoubleComplex(sqrt(cuCreal(temp)), 0.0);
        }
      }

      __syncthreads();

      // If info != 0 return (matrix is not positive definite)
      if (sinfo != 0)
        return;

      // ZSSCAL
      if (threadIdx.x > j)
        a[lower<bx>(threadIdx.x, j)] = cuDiva(temp, cuCreal(a[lower<bx>(j, j)]));

      __syncthreads();
    }

    // Write the lower triangle of A back to global memory
    for (int j = 0; j < n; j++) {
      if (threadIdx.x >= j)
        A[j * lda + threadIdx.x] = a[lower<bx>(threadIdx.x, j)];
    }
  }
}

template void zpotf2<CBlasUpper, 32>(int, cuDoubleComplex *, int, int *);
template void zpotf2<CBlasLower, 32>(int, cuDoubleComplex *, int, int *);
