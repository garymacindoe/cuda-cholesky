#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include "config.h"
#include "handle.h"

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

static const float zero = 0.0f;
static const float one = 1.0f;

static inline void strti2(CBlasUplo uplo, CBlasDiag diag,
                          size_t n,
                          float * restrict A, size_t lda,
                          long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register float ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = one / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -one;

      for (size_t k = 0; k < j; k++) {
        register float temp = A[j * lda + k];
        if (diag == CBlasNonUnit) A[j * lda + k] *= A[k * lda + k];
        for (size_t i = 0; i < k; i++)
          A[j * lda + i] += temp * A[k * lda + i];
      }
      for (size_t i = 0; i < j; i++)
        A[j * lda + i] *= ajj;
    }
  }
  else {
    size_t j = n - 1;
    do {
      register float ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = one / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -one;

      for (size_t i = n - 1; i > j; i--) {
        register float temp = A[j * lda + i];
        if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + k] += temp * A[i * lda + k];
      }
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] *= ajj;
    } while (j-- > 0);
  }
}

static inline void strti22(CBlasUplo uplo, CBlasDiag diag,
                           size_t n,
                           const float * restrict A, size_t lda,
                           float * restrict B, size_t ldb,
                           long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register float bjj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        B[j * ldb + j] = one / A[j * lda + j];
        bjj = -B[j * ldb + j];
      }
      else
        bjj = -one;

      for (size_t i = 0; i < j; i++)
        B[j * ldb + i] = A[j * lda + i];
      for (size_t k = 0; k < j; k++) {
        register float temp = B[j * ldb + k];
        if (diag == CBlasNonUnit) B[j * ldb + k] *= B[k * ldb + k];
        for (size_t i = 0; i < k; i++)
          B[j * ldb + i] += temp * B[k * ldb + i];
      }
      for (size_t i = 0; i < j; i++)
        B[j * ldb + i] *= bjj;
    }
  }
  else {
    size_t j = n - 1;
    do {
      register float bjj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        B[j * ldb + j] = one / A[j * lda + j];
        bjj = -B[j * ldb + j];
      }
      else
        bjj = -one;

      for (size_t i = j + 1; i < n; i++)
        B[j * ldb + i] = A[j * lda + i];
      for (size_t i = n - 1; i > j; i--) {
        register float temp = B[j * ldb + i];
        if (diag == CBlasNonUnit) B[j * ldb + i] *= B[i * ldb + i];
        for (size_t k = i + 1; k < n; k++)
          B[j * ldb + k] += temp * B[i * ldb + k];
      }
      for (size_t i = j + 1; i < n; i++)
        B[j * ldb + i] *= bjj;
    } while (j-- > 0);
  }
}

void strtri(CBlasUplo uplo, CBlasDiag diag,
            size_t n,
            float * restrict A, size_t lda,
            long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  const size_t nb = (uplo == CBlasUpper) ? 32 : 64;

  if (n < nb) {
    strti2(uplo, diag, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);
      strmm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag,
            j, jb,
            one, A, lda, &A[j * lda], lda);
      strsm(CBlasRight, CBlasUpper, CBlasNoTrans, diag,
            j, jb,
            -one, &A[j * lda + j], lda,
            &A[j * lda], lda);
      strti2(CBlasUpper, diag,
             jb,
             &A[j * lda + j], lda,
             info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    }
  }
  else {
    size_t j = (n + nb - 1) & ~(nb - 1);
    do {
      j -= nb;
      const size_t jb = min(nb, n - j);
      if (j + jb < n) {
        strmm(CBlasLeft, CBlasLower, CBlasNoTrans, diag,
              n - j - jb, jb,
              one, &A[(j + jb) * lda + j + jb], lda, &A[j * lda + j + jb], lda);
        strsm(CBlasRight, CBlasLower, CBlasNoTrans, diag,
              n - j - jb, jb,
              -one, &A[j * lda + j], lda,
              &A[j * lda + j + jb], lda);
      }
      strti2(CBlasLower, diag,
             jb,
             &A[j * lda + j], lda,
             info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    } while (j > 0);
  }
}

void strtri2(CBlasUplo uplo, CBlasDiag diag,
             size_t n,
             const float * restrict A, size_t lda,
             float * restrict B, size_t ldb,
             long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (ldb < n)
    *info = -7;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  const size_t nb = (uplo == CBlasUpper) ? 32 : 64;

  if (n < nb) {
    strti22(uplo, diag, n, A, lda, B, ldb, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);
      strmm2(CBlasLeft, CBlasUpper, CBlasNoTrans, diag,
             j, jb,
             one, B, ldb, &A[j * lda], lda,
             &B[j * ldb], ldb);
      strsm(CBlasRight, CBlasUpper, CBlasNoTrans, diag,
            j, jb,
            -one, &A[j * lda + j], lda,
            &B[j * ldb], ldb);
      strti22(CBlasUpper, diag,
              jb,
              &A[j * lda + j], lda,
              &B[j * ldb + j], ldb,
              info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    }
  }
  else {
    size_t j = (n + nb - 1) & ~(nb - 1);
    do {
      j -= nb;
      const size_t jb = min(nb, n - j);
      if (j + jb < n) {
        strmm2(CBlasLeft, CBlasLower, CBlasNoTrans, diag,
               n - j - jb, jb,
               one, &B[(j + jb) * ldb + j + jb], ldb, &A[j * lda + j + jb], lda,
               &B[j * ldb + j + jb], ldb);
        strsm(CBlasRight, CBlasLower, CBlasNoTrans, diag,
              n - j - jb, jb,
              -one, &A[j * lda + j], lda,
              &B[j * ldb + j + jb], ldb);
      }
      strti22(CBlasLower, diag,
              jb,
              &A[j * lda + j], lda,
              &B[j * ldb + j], ldb,
              info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    } while (j > 0);
  }
}

CUresult cuStrtri(CULAPACKhandle handle,
                  CBlasUplo uplo, CBlasDiag diag,
                  size_t n,
                  CUdeviceptr A, size_t lda,
                  long * info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  float * D;
  CUdeviceptr X;
  size_t ldb, ldx;
  CUstream stream0, stream1;

  // (Maximum) dynamic block size
  size_t nb = n / 2;

  // Allocate page-locked host memory for diagonal block
  CU_ERROR_CHECK(cuMemAllocHost((void **)&D, (ldb = (n + 3u) & ~3u) * nb * sizeof(float)));

  // Allocate a temporary column for the out of place matrix multiply
  CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(float), nb, sizeof(float)));
  ldx /= sizeof(float);

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));

  if (uplo == CBlasUpper) {
    // Increase block size
    for (size_t j = 0, nb = 1; j < n; j += nb, nb = min(n / 2, nb * 2)) {
      const size_t jb = min(nb, n - j);

      // Work out whether it is worthwhile to do block column copy for the block size
      const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
      const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
      const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

      const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
      const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
      // Can only copy column back if the column was copied in the first place
      const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

      /* Multiply the current column by the big square matrix to the left and
       * store the result in X */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle,
                              CBlasLeft, CBlasUpper, CBlasNoTrans, diag, j, jb,
                              one, A, lda, A + j * lda * sizeof(float), lda,
                              X, ldx, stream0));

      /* Start copying diagonal block onto host asynchronously */
      float * B;
      if (bcc_dtoh) {
        // Copy the entire column in one go
        CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + j * lda * sizeof(float), n * jb * sizeof(float), stream1));
        B = &D[j];    // The diagonal block is half-way down
      }
      else {
        // Copy each column of the diagonal block separately
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, j, j,
                                          jb, jb, sizeof(float), stream1));
        B = D;      // The diagonal block is at the top of the column
      }

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the inverse of the diagonal block using the CPU */
      strtri(uplo, diag, jb, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      if (bcc_htod)
        CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), D, n * jb * sizeof(float), stream1));
      else
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                          jb, jb, sizeof(float), stream1));

      if (*info != 0) {
        *info += (long)j;
        break;
      }

      /* Wait until the diagonal block has been copied back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));

      /* Multiply the temporary column by the small square matrix on the
       * diagonal below and store the result back in the correct place in A */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle,
                              CBlasRight, CBlasUpper, CBlasNoTrans, diag, j, jb,
                              -one, A + (j * lda + j) * sizeof(float), lda, X, ldx,
                              A + j * lda * sizeof(float), lda, stream0));
    }
  }
  else {
    for (size_t i = 0, nb = 1; i < n; i += nb, nb = min(n / 2, nb * 2)) {
      const size_t jb = min(nb, n - i);
      const size_t j = n - i - jb;

      // Work out whether it is worthwhile to do block column copy for the block size
      const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
      const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
      const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

      const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
      const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
      // Can only copy column back if the column was copied in the first place
      const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

      /* Multiply the current column by the big square matrix to the right and
       * store the result in X */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle,
                              CBlasLeft, CBlasLower, CBlasNoTrans, diag, n - j - jb, jb,
                              one, A + ((j + jb) * lda + j + jb) * sizeof(float), lda,
                              A + (j * lda + j + jb) * sizeof(float), lda, X, ldx, stream0));

      /* Start copying diagonal block onto host asynchronously */
      float * B;
      if (bcc_dtoh) {
        // Copy the entire column in one go
        CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + j * lda * sizeof(float), n * jb * sizeof(float), stream1));
        B = &D[j];    // The diagonal block is half-way down
      }
      else {
        // Copy each column of the diagonal block separately
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, j, j,
                                          jb, jb, sizeof(float), stream1));
        B = D;      // The diagonal block is at the top of the column
      }

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the inverse of the diagonal block using the CPU */
      strtri(uplo, diag, jb, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      if (bcc_htod)
        CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), D, n * jb * sizeof(float), stream1));
      else
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                          jb, jb, sizeof(float), stream1));

      if (*info != 0) {
        *info += (long)j;
        break;
      }

      /* Wait until the diagonal block has been copied back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));

      /* Multiply the temporary column by the small square matrix on the
       * diagonal above and store the result back in the correct place in A */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle,
                              CBlasRight, CBlasLower, CBlasNoTrans, diag, n - j - jb, jb,
                              -one, A + (j * lda + j) * sizeof(float), lda, X, ldx,
                              A + (j * lda + j + jb) * sizeof(float), lda, stream0));

    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(D));
  CU_ERROR_CHECK(cuMemFree(X));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUStrtri(CUmultiGPULAPACKhandle handle,
                          CBlasUplo uplo, CBlasDiag diag,
                          size_t n,
                          float * restrict A, size_t lda,
                          long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  if (uplo == CBlasUpper) {
    const size_t nb = SGEMM_N_MB;

    // Upper triangular STRTRI
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);
      CU_ERROR_CHECK(cuMultiGPUStrmm(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasNoTrans, diag,
                                     j, jb, one, A, lda, &A[j * lda], lda));
      CU_ERROR_CHECK(cuMultiGPUStrsm(handle->blas_handle, CBlasRight, CBlasUpper, CBlasNoTrans, diag,
                                     j, jb, -one, &A[j * lda + j], lda, &A[j * lda], lda));
      strtri(CBlasUpper, diag, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
    }
  }
  else {
    // Lower triangular STRTRI
    const size_t nb = SGEMM_N_MB;
    const size_t r = n % nb;
    size_t j = (r == 0) ? n : n + nb - r;
    do {
      j -= nb;
      const size_t jb = min(nb, n - j);
      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUStrmm(handle->blas_handle, CBlasLeft, CBlasLower, CBlasNoTrans, diag,
                                       n - j - jb, jb,
                                       one, &A[(j + jb) * lda + j + jb], lda,
                                       &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handle->blas_handle, CBlasRight, CBlasLower, CBlasNoTrans, diag,
                                       n - j - jb, jb,
                                       -one, &A[j * lda + j], lda,
                                       &A[j * lda + j + jb], lda));
      }
      strtri(CBlasLower, diag, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
    } while (j > 0);
  }

  return CUDA_SUCCESS;
}

