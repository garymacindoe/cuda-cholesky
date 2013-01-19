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

static inline void slauu2(CBlasUplo uplo,
                          size_t n,
                          float * restrict A, size_t lda) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register float aii = A[i * lda + i];
      register float temp = zero;
      for (size_t k = i; k < n; k++)
        temp += A[k * lda + i] * A[k * lda + i];
      A[i * lda + i] = temp;

      for (size_t k = 0; k < i; k++)
        A[i * lda + k] *= aii;
      for (size_t j = i + 1; j < n; j++) {
        register float temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          A[i * lda + k] += temp * A[j * lda + k];
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i++) {
      register float aii = A[i * lda + i];
      register float temp = zero;
      for (size_t k = i; k < n; k++)
        temp += A[i * lda + k] * A[i * lda + k];
      A[i * lda + i] = temp;

      for (size_t k = 0; k < i; k++)
        A[k * lda + i] *= aii;
      for (size_t k = 0; k < i; k++) {
        register float temp = zero;
        for (size_t j = i + 1; j < n; j++)
          temp += A[k * lda + j] * A[i * lda + j];
        A[k * lda + i] += temp;
      }
    }
  }
}

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

  strtri(uplo, CBlasNonUnit, n, A, lda, info);

  if (*info > 0)
    return;

  const size_t nb = 64;

  if (n < nb) {
    slauu2(uplo, n, A, lda);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      strmm(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
            one, &A[i * lda + i], lda, &A[i * lda], lda);
      slauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        sgemm(CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
              one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda], lda);
        ssyrk(CBlasUpper, CBlasNoTrans, ib, n - i - ib,
              one, &A[(i + ib) * lda + i], lda, one, &A[i * lda + i], lda);
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      strmm(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
            one, &A[i * lda + i], lda, &A[i], lda);
      slauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        sgemm(CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
              one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
              one, &A[i], lda);
        ssyrk(CBlasLower, CBlasTrans, ib, n - i - ib,
              one, &A[i * lda + i + ib], lda, one, &A[i * lda + i], lda);
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
