#include "lapack.h"
#include "error.h"
#include <stdio.h>
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

static inline void dlauu2(CBlasUplo uplo,
                          size_t n,
                          double * restrict A, size_t lda) {
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register double ajj = A[j * lda + j];
      for (size_t i = 0; i <= j; i++)
        A[j * lda + i] *= ajj;

      for (size_t k = j + 1; k < n; k++) {
        register double temp = A[k * lda + j];
        for (size_t i = 0; i <= j; i++)
          A[j * lda + i] += temp * A[k * lda + i];
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        A[j * lda + i] *= A[i * lda + i];

        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += A[i * lda + k] * A[j * lda + k];
      }
    }
  }
}

void dlauum(CBlasUplo uplo,
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

  if (n == 0)
    return;

  const size_t nb = (uplo == CBlasUpper) ? 16 : 32;

  if (nb > n) {
    dlauu2(uplo, n, A, lda);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      dtrmm(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
            one, &A[i * lda + i], lda, &A[i * lda], lda);
      dlauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        dgemm(CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
              one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda], lda);
        dsyrk(CBlasUpper, CBlasNoTrans, ib, n - i - ib,
              one, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda + i], lda);
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      dtrmm(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
            one, &A[i * lda + i], lda, &A[i], lda);
      dlauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        dgemm(CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
              one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
              one, &A[i], lda);
        dsyrk(CBlasLower, CBlasTrans, ib, n - i - ib,
              one, &A[i * lda + i + ib], lda,
              one, &A[i * lda + i], lda);
      }
    }
  }
}

CUresult cuDlauum(CULAPACKhandle handle,
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

  double * D;
  CUdeviceptr X;
  size_t ldb, ldx;
  CUstream stream0, stream1;

  // (Maximum) dynamic block size
  size_t nb = n / 4;

  // Allocate page-locked host memory for diagonal block
  CU_ERROR_CHECK(cuMemAllocHost((void **)&D, (ldb = (n + 1u) & ~1u) * nb * sizeof(double)));

  // Create streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));

  if (uplo == CBlasUpper) {
    // Allocate a temporary column for the out of place matrix multiply
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(double), nb, sizeof(double)));
    ldx /= sizeof(double);

    // Decrease block size towards centre then increase
    for (size_t i = 0; i < n; i += nb, nb = (i < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t ib = min(nb, n - i);

      // Work out whether it is worthwhile to do block column copy for the block size
      const double column_dtoh = (double)(n * ib * sizeof(double)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
      const double block_dtoh = (double)ib * ((double)(ib * sizeof(double)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
      const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuDtrmm2(handle->blas_handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
                              one, A + (i * lda + i) * sizeof(double), lda,
                              A + i * lda * sizeof(double), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuDgemm2(handle->blas_handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                              one, A + (i + ib) * lda * sizeof(double), lda,
                              A + ((i + ib) * lda + i) * sizeof(double), lda,
                              one, X, ldx, A + i * lda * sizeof(double), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      double * B;
      if (bcc_dtoh) {
        // Copy the entire column in one go
        CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + i * lda * sizeof(double), n * ib * sizeof(double), stream0));
        B = &D[i];    // The diagonal block is half-way down
      }
      else {
        // Copy each column of the diagonal block separately
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, i, i,
                                          ib, ib, sizeof(double), stream0));
        B = D;      // The diagonal block is at the top of the column
      }

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Form the multiplication of the diagonal block using the CPU */
      dlauum(uplo, ib, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(double), stream0));

      /* Perform the DSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuDsyrk(handle->blas_handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                             one, A + ((i + ib) * lda + i) * sizeof(double), lda,
                             one, A + (i * lda + i) * sizeof(double), lda, stream0));
    }
  }
  else {
    // Allocate a temporary row for the out of place matrix multiply
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, nb * sizeof(double), n, sizeof(double)));
    ldx /= sizeof(double);

    // Decrease block size towards centre then increase
    for (size_t i = 0; i < n; i += nb, nb = (i < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t ib = min(nb, n - i);

      // Work out whether it is worthwhile to do block column copy for the block size
      const double column_dtoh = (double)(n * ib * sizeof(double)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
      const double block_dtoh = (double)ib * ((double)(ib * sizeof(double)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
      const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

      const double column_htod = (double)(n * ib * sizeof(double)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
      const double block_htod = (double)ib * ((double)(ib * sizeof(double)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
      // Can only copy column back if the column was copied in the first place
      const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuDtrmm2(handle->blas_handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                              one, A + (i * lda + i) * sizeof(double), lda,
                              A + i * sizeof(double), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuDgemm2(handle->blas_handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                              one, A + (i * lda + i + ib) * sizeof(double), lda,
                              A + (i + ib) * sizeof(double), lda,
                              one, X, ldx, A + i * sizeof(double), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      double * B;
      if (bcc_dtoh) {
        // Copy the entire column in one go
        CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + i * lda * sizeof(double), n * ib * sizeof(double), stream0));
        B = &D[i];    // The diagonal block is half-way down
      }
      else {
        // Copy each column of the diagonal block separately
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, i, i,
                                          ib, ib, sizeof(double), stream0));
        B = D;      // The diagonal block is at the top of the column
      }

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Form the multiplication of the diagonal block using the CPU */
      dlauum(uplo, ib, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      if (bcc_htod)   // Only works for lower triangular dlauum
        CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + i * lda * sizeof(double), D, n * ib * sizeof(double), stream0));
      else
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                          ib, ib, sizeof(double), stream0));

      /* Perform the DSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuDsyrk(handle->blas_handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                             one, A + (i * lda + i + ib) * sizeof(double), lda,
                             one, A + (i * lda + i) * sizeof(double), lda, stream1));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(D));
  CU_ERROR_CHECK(cuMemFree(X));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDlauum(CUmultiGPULAPACKhandle handle,
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

  const size_t nb = (uplo == CBlasUpper) ? DGEMM_N_MB : DGEMM_T_MB;

  if (n < nb) {
    dlauum(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      CU_ERROR_CHECK(cuMultiGPUDtrmm(handle->blas_handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit,
                                     i, ib, one, &A[i * lda + i], lda, &A[i * lda], lda));
      dlauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                                       one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
                                       one, &A[i * lda], lda));
        CU_ERROR_CHECK(cuMultiGPUDsyrk(handle->blas_handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                                       one, &A[(i + ib) * lda + i], lda, one, &A[i * lda + i], lda));
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      CU_ERROR_CHECK(cuMultiGPUDtrmm(handle->blas_handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                                     one, &A[i * lda + i], lda, &A[i], lda));
      dlauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
                                       one, &A[i], lda));
        CU_ERROR_CHECK(cuMultiGPUDsyrk(handle->blas_handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, one, &A[i * lda + i], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
