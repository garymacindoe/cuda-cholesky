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
  size_t ldb;
  CUstream stream0, stream1;

  // (Maximum) dynamic block size
  size_t nb = n / 4;

  // Allocate page-locked host memory for diagonal block
  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 3u) & ~3u) * nb * sizeof(float)));

  // Create streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));

  if (uplo == CBlasUpper) {
    // Decrease block size towards centre then increase
    for (size_t i = 0; i < n; i += nb, nb = (i < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm(handle->blas_handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
                             one, A + (i * lda + i) * sizeof(float), lda,
                             A + i * lda * sizeof(float), lda, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                             one, A + (i + ib) * lda * sizeof(float), lda,
                             A + ((i + ib) * lda + i) * sizeof(float), lda,
                             one, A + i * lda * sizeof(float), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                        ib, ib, sizeof(float), stream0));

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Form the multiplication of the diagonal block using the CPU */
      slauum(uplo, ib, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(float), stream0));

      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                             one, A + ((i + ib) * lda + i) * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream0));
    }
  }
  else {
    // Decrease block size towards centre then increase
    for (size_t i = 0; i < n; i += nb, nb = (i < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm(handle->blas_handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                             one, A + (i * lda + i) * sizeof(float), lda,
                             A + i * sizeof(float), lda, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                             one, A + (i * lda + i + ib) * sizeof(float), lda,
                             A + (i + ib) * sizeof(float), lda,
                             one, A + i * sizeof(float), lda, stream1));

      /* Start copying diagonal block onto host asynchronously */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                        ib, ib, sizeof(float), stream0));

      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Form the multiplication of the diagonal block using the CPU */
      slauum(uplo, ib, B, ldb, info);

      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(float), stream0));

      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                             one, A + (i * lda + i + ib) * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream1));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));

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

  const size_t nb = (uplo == CBlasUpper) ? SGEMM_N_MB : SGEMM_T_MB;

  if (n < nb) {
    slauum(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
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
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

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
