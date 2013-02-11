#include "lapack.h"
#include "handle.h"
#include "error.h"
#include <stdio.h>
#include "config.h"
#include "slauum.fatbin.c"

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

static inline CUresult cuMemcpyDtoD2DAsync(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                                           CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                                           size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static const float zero = 0.0f;
static const float one = 1.0f;

static inline void slauu2(CBlasUplo uplo,
                          size_t n,
                          float * restrict A, size_t lda) {
  if (uplo == CBlasUpper) {
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
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        A[j * lda + i] *= A[i * lda + i];

        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += A[i * lda + k] * A[j * lda + k];
      }
    }
  }
}

void slauum(CBlasUplo uplo,
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

  const size_t nb = (uplo == CBlasUpper) ? 16 : 32;

  if (nb > n) {
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
              one, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda + i], lda);
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
              one, &A[i * lda + i + ib], lda,
              one, &A[i * lda + i], lda);
      }
    }
  }
}

static inline CUresult cuSlauu2(CULAPACKhandle handle, CBlasUplo uplo,
                                size_t n,
                                CUdeviceptr A, size_t lda, CUstream stream) {
  const unsigned int bx = 64;
  if (n > bx)
    return CUDA_ERROR_INVALID_VALUE;

  if (handle->slauum == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->slauum, imageBytes));

  char name[37];
  snprintf(name, 37, "_Z6slauu2IL9CBlasUplo%dELj%uEEvPfii", uplo, bx);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->slauum, name));

  void * params[] = { &A, &lda, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

static CUresult hybridSlauum(CBlasUplo uplo,
                             CUdeviceptr A, size_t lda, float * X, size_t ldb,
                             size_t i, size_t ib, size_t n, long * info, CUstream stream) {

  // Work out whether it is worthwhile to do block column copy for the block size
  const double column_dtoh = (double)(n * ib * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
  const double block_dtoh = (double)ib * ((double)(ib * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
  const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

  const double column_htod = (double)(n * ib * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
  const double block_htod = (double)ib * ((double)(ib * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
  // Can only copy column back if the column was copied in the first place
  const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

  /* Start copying diagonal block onto host asynchronously */
  float * B;
  if (bcc_dtoh) {
    // Copy the entire column in one go
    CU_ERROR_CHECK(cuMemcpyDtoHAsync(X, A + i * lda * sizeof(float), n * ib * sizeof(float), stream));
    B = &X[i];    // The diagonal block is half-way down
  }
  else {
    // Copy each column of the diagonal block separately
    CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(X, ldb, 0, 0, A, lda, i, i,
                                      ib, ib, sizeof(float), stream));
    B = X;      // The diagonal block is at the top of the column
  }

  /* Wait until the diagonal block has been copied */
  CU_ERROR_CHECK(cuStreamSynchronize(stream));
  /* Form the multiplication of the diagonal block using the CPU */
  slauum(uplo, ib, B, ldb, info);

  /* Copy the diagonal block back onto the device */
  if (uplo == CBlasLower && bcc_htod)   // Only works for lower triangular slauum
    CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + i * lda * sizeof(float), X, n * ib * sizeof(float), stream));
  else
    CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                      ib, ib, sizeof(float), stream));

  return CUDA_SUCCESS;
}

CUresult cuSlauum(CULAPACKhandle handle,
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

  float * B;
  CUdeviceptr D;
  size_t ldb, ldd;
  CUstream stream0, stream1;

  /**
   * In both loops for STRTRI and SLAUUM A is updated column by column whether
   * upper or lower triangular.  The STRMM consumes most of the FLOPS in STRTRI
   * while the SGEMM consumes most of the FLOPS in SLAUUM.  STRMM is always
   * called with transA == CBlasNoTrans as is SGEMM except in the lower
   * triangular SLAUUM.  This means that the size of B in host memory changes
   * between loops when A is lower triangular.
   */
  const size_t nb = 512;

  // Allocate page-locked host memory for diagonal block column
  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (n + 3u) & ~3u) * nb * sizeof(float)));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    // Upper triangular requires a temporary column for out of place STRMM
    CU_ERROR_CHECK(cuMemAllocPitch(&D, &ldd, n * sizeof(float), nb, sizeof(float)));
    ldd /= sizeof(float);

    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
                              one, A + (i * lda + i) * sizeof(float), lda,
                              A + i * lda * sizeof(float), lda, D, ldd, stream0));
      /* Ensure the STRMM has finished before starting the SGEMM on a different stream */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                              one, A + (i + ib) * lda * sizeof(float), lda,
                              A + ((i + ib) * lda + i) * sizeof(float), lda,
                              one, D, ldd, A + i * lda * sizeof(float), lda, stream1));
      CU_ERROR_CHECK(hybridSlauum(uplo, A, lda, B, ldb, i, ib, n, info, stream0));
      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                             one, A + ((i + ib) * lda + i) * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream0));
      /* Ensure the SSYRK has finished before starting the STRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
    }
  }
  else {
    // Lower triangular requires a temporary row for out of place STRMM
    CU_ERROR_CHECK(cuMemAllocPitch(&D, &ldd, nb * sizeof(float), n, sizeof(float)));
    ldd /= sizeof(float);

    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                              one, A + (i * lda + i) * sizeof(float), lda,
                              A + i * sizeof(float), lda, D, ldd, stream0));
      /* Ensure the STRMM has finished before starting the SGEMM on a different stream */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                              one, A + (i * lda + i + ib) * sizeof(float), lda,
                              A + (i + ib) * sizeof(float), lda,
                              one, D, ldd, A + i * sizeof(float), lda, stream1));
      CU_ERROR_CHECK(hybridSlauum(uplo, A, lda, B, ldb, i, ib, n, info, stream0));
      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                             one, A + (i * lda + i + ib) * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream1));
      /* Ensure the SSYRK has finished before starting the STRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));
  CU_ERROR_CHECK(cuMemFree(D));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSlauum(CUmultiGPULAPACKhandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          float * restrict A, size_t lda,
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

  if (uplo == CBlasUpper) {
    const size_t nb = SGEMM_N_MB;

    // Upper triangular SLAUUM
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      CU_ERROR_CHECK(cuMultiGPUStrmm(handle->blas_handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit,
                                     i, ib, one, &A[i * lda + i], lda, &A[i * lda], lda));
      slauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                                       one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
                                       one, &A[i * lda], lda));
        CU_ERROR_CHECK(cuMultiGPUSsyrk(handle->blas_handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                                       one, &A[(i + ib) * lda + i], lda, one, &A[i * lda + i], lda));
      }
    }
  }
  else {
    const size_t mb = SGEMM_T_MB;

    // Lower triangular SLAUUM
    for (size_t i = 0; i < n; i += mb) {
      const size_t ib = min(mb, n - i);

      CU_ERROR_CHECK(cuMultiGPUStrmm(handle->blas_handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                                     one, &A[i * lda + i], lda, &A[i], lda));
      slauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
                                       one, &A[i], lda));
        CU_ERROR_CHECK(cuMultiGPUSsyrk(handle->blas_handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, one, &A[i * lda + i], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
