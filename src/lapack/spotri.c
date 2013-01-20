#include "lapack.h"
#include "error.h"
// #include <stdio.h>
#include <math.h>
#include "../blas/handle.h"

static inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }

static inline CUresult cuMemcpyHtoD2DAsync(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                                          const void * B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static inline CUresult cuMemcpyDtoH2DAsync(void * A, size_t lda, size_t ai, size_t aj,
                                          CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_HOST, A, 0, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static const float zero = 0.0f;
static const float one = 1.0f;

void spotri(CBlasUplo uplo,
            size_t n,
            float * restrict A, size_t lda,
            long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      if (A[j * lda + j] == zero) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = one / A[j * lda + j];
      register float ajj = -A[j * lda + j];

      for (size_t k = 0; k < j; k++) {
        register float temp = A[j * lda + k];
        A[j * lda + k] *= A[k * lda + k];
        for (size_t i = 0; i < k; i++)
          A[j * lda + i] += temp * A[k * lda + i];
      }
      for (size_t i = 0; i < j; i++)
        A[j * lda + i] *= ajj;
    }

    for (size_t j = 0; j < n; j++) {
      register float ajj = A[j * lda + j];
      for (size_t i = 0; i <= j; i++)
        A[j * lda + i] *= ajj;
      for (size_t k = j + 1; k < n; k++) {
        register float temp = A[k * lda + j];
        for (size_t i = 0; i <= j; i++)
          A[j * lda + i] += temp * A[k * lda + i];
      }
    }
  }
  else {
    size_t j = n - 1;
    do {
      if (A[j * lda + j] == zero) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = one / A[j * lda + j];
      register float ajj = -A[j * lda + j];

      size_t i = n - 1;
      do {
        register float temp = A[j * lda + i];
        A[j * lda + i] *= A[i * lda + i];
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + k] += temp * A[i * lda + k];
      } while (i-- > j);
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] *= ajj;
    } while (j-- > 0);

    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        A[j * lda + i] *= A[i * lda + i];
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += A[i * lda + k] * A[j * lda + k];
      }
    }
  }
}

CUresult cuSpotri(CBlasUplo uplo,
                  size_t n,
                  CUdeviceptr A, size_t lda,
                  long * info) {
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSpotri(CUmultiGPUBlasHandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          float * restrict A, size_t lda,
                          long * restrict info) {
  return CUDA_SUCCESS;
}
