#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "handle.h"
#include "config.h"

static inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }
static inline size_t max(size_t a, size_t b) { return (a > b) ? a : b; }

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

static const double zero = 0.0;
static const double one = 1.0;

static inline void dpotf2(CBlasUplo uplo,
                          size_t n,
                          double * restrict A, size_t lda,
                          long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register double temp = zero;
      const double * restrict B = A;
      for (size_t k = 0; k < i; k++)
        temp += A[i * lda + k] * B[i * lda + k];

      register double aii = A[i * lda + i] - temp;
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrt(aii);
      A[i * lda + i] = aii;

      for (size_t j = i + 1; j < n; j++) {
        temp = zero;
        for (size_t k = 0; k < i; k++)
          temp += A[j * lda + k] * A[i * lda + k];
        A[j * lda + i] = (A[j * lda + i] - temp) / aii;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < j; k++) {
        register double temp = A[k * lda + j];
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      register double ajj = A[j * lda + j];
      if (ajj <= zero || isnan(ajj)) {
        *info = (long)j + 1;
        return;
      }
      ajj = sqrt(ajj);
      A[j * lda + j] = ajj;
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

void dpotrf(CBlasUplo uplo,
            size_t n,
            double * restrict A, size_t lda,
            long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  const size_t nb = (uplo == CBlasUpper) ? 16 : 32;

  if (nb > n) {
    dpotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      dsyrk(CBlasUpper, CBlasTrans, jb, j,
            -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      dpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        dgemm(CBlasTrans, CBlasNoTrans, jb, n - j - jb, j,
              -one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
              one, &A[(j + jb) * lda + j], lda);
        dtrsm(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb,
              one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      dsyrk(CBlasLower, CBlasNoTrans, jb, j,
            -one, &A[j], lda,one, &A[j * lda + j], lda);
      dpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        dgemm(CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
              -one, &A[j + jb], lda, &A[j], lda,
              one, &A[j * lda + j + jb], lda);
        dtrsm(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb,
              one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

CUresult cuDpotrf(CULAPACKhandle handle,
                  CBlasUplo uplo,
                  size_t n,
                  CUdeviceptr A, size_t lda,
                  long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  // Block size
  const size_t nb = (uplo == CBlasUpper) ? 256 : 128;

  double * B;
  size_t ldb;
  CUstream stream0, stream1;

  // Allocate memory for diagonal block on host
  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 1u) & ~1u) * nb * sizeof(double)));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuDsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                             -one, A + j * lda * sizeof(double), lda,
                             one, A + (j * lda + j) * sizeof(double), lda, stream0));

      /* Matrix multiply using column above and matrix to the right to update
       * the row to the right of the diagonal block */
      CU_ERROR_CHECK(cuDgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans,
                             jb, n - j - jb, j,
                             -one, A + j * lda * sizeof(double), lda,
                             A + (j + jb) * lda * sizeof(double), lda,
                             one, A + ((j + jb) * lda + j) * sizeof(double), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(double), stream0));

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));

      /* Perform the diagonal block decomposition using the CPU */
      dpotrf(uplo, jb, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(double), stream0));

      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }

      /* Triangular solve to update the row to the right using the diagonal
       * block */
      CU_ERROR_CHECK(cuDtrsm(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit,
                             jb, n - j - jb, one, A + (j * lda + j) * sizeof(double), lda,
                             A + ((j + jb) * lda + j) * sizeof(double), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuDsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                             -one, A + j * sizeof(double), lda,
                             one, A + (j * lda + j) * sizeof(double), lda, stream0));

      /* Matrix multiply using row to the left and matrix below to update the
       * column below the diagonal block */
      CU_ERROR_CHECK(cuDgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                             -one, A + (j + jb) * sizeof(double), lda,
                             A + j * sizeof(double), lda,
                             one, A + (j * lda + j + jb) * sizeof(double), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(double), stream0));

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));

      /* Perform the diagonal block decomposition using the CPU */
      dpotrf(uplo, jb, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                        jb, jb, sizeof(double), stream0));

      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }

      /* Triangular solve to update the column below using the diagonal block */
      CU_ERROR_CHECK(cuDtrsm(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit,
                             n - j - jb, jb, one, A + (j * lda + j) * sizeof(double), lda,
                             A + (j * lda + j + jb) * sizeof(double), lda, stream0));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDpotrf(CUmultiGPULAPACKhandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          double * restrict A, size_t lda,
                          long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  const size_t nb = (uplo == CBlasUpper) ? DGEMM_T_MB : DGEMM_N_NB;

  if (n < nb) {
    dpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUDsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                                     -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle->blas_handle));
      dpotrf(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j,
                                       -one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
                                       one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUDtrsm(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb,
                                       one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUDsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                                     -one, &A[j], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle->blas_handle));
      dpotrf(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                                       -one, &A[j + jb], lda, &A[j], lda,
                                       one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUDtrsm(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb,
                                       one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}