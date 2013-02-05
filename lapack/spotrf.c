#include "lapack.h"
#include "handle.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "config.h"
#include "spotrf.fatbin.c"

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

static inline void spotf2(CBlasUplo uplo,
                          size_t n,
                          float * restrict A, size_t lda,
                          long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register float temp = zero;
      const float * restrict B = A;
      for (size_t k = 0; k < i; k++)
        temp += A[i * lda + k] * B[i * lda + k];

      register float aii = A[i * lda + i] - temp;
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrtf(aii);
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
        register float temp = A[k * lda + j];
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      register float ajj = A[j * lda + j];
      if (ajj <= zero || isnan(ajj)) {
        *info = (long)j + 1;
        return;
      }
      ajj = sqrtf(ajj);
      A[j * lda + j] = ajj;
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

void spotrf(CBlasUplo uplo,
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

  if (n == 0) return;

  const size_t nb = (uplo == CBlasUpper) ? 16 : 32;

  if (nb > n) {
    spotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      ssyrk(CBlasUpper, CBlasTrans, jb, j,
            -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      spotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        sgemm(CBlasTrans, CBlasNoTrans, jb, n - j - jb, j,
              -one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
              one, &A[(j + jb) * lda + j], lda);
        strsm(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb,
              one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      ssyrk(CBlasLower, CBlasNoTrans, jb, j,
            -one, &A[j], lda,one, &A[j * lda + j], lda);
      spotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        sgemm(CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
              -one, &A[j + jb], lda, &A[j], lda,
              one, &A[j * lda + j + jb], lda);
        strsm(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb,
              one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

static inline CUresult cuSpotf2(CULAPACKhandle handle, CBlasUplo uplo,
                                size_t n,
                                CUdeviceptr A, size_t lda,
                                CUdeviceptr info, CUstream stream) {
  const unsigned int bx = 64;
  if (n > bx)
    return CUDA_ERROR_INVALID_VALUE;

  if (handle->spotrf == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->spotrf, imageBytes));

  char name[39];
  snprintf(name, 39, "_Z6spotf2IL9CBlasUplo%dELj%uEEvPfPiii", uplo, bx);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->spotrf, name));

  void * params[] = { &A, &info, &lda, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuSpotrf(CULAPACKhandle handle, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  /**
   * The SGEMM consumes most of the FLOPs in the Cholesky decomposition so the
   * block sizes are chosen to favour it.  In the upper triangular case it is
   * the row matrix to the right of the diagonal block that is updated via
   * D = -A^T * C + D (i.e. the A argument to SGEMM is transposed) therefore the
   * block size is SGEMM_T_MB.  For the lower triangular case it is the column
   * matrix below the diagonal block that is updated via D = -C * A^T + D so the
   * block size is SGEMM_N_NB.
   */
  const size_t nb = 64;//(uplo == CBlasUpper) ? SGEMM_T_MB : SGEMM_N_NB;

  float * B;
  size_t ldb;
  CUstream stream0, stream1;

  // Allocate page-locked host memory for diagonal block
  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 3u) & ~3u) * nb * sizeof(float)));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                             -one, A + j * lda * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));
      /* Overlap the SSYRK with an SGEMM (on a different stream) */
      CU_ERROR_CHECK(cuSgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j,
                             -one, A + j * lda * sizeof(float), lda,
                             A + (j + jb) * lda * sizeof(float), lda,
                             one, A + ((j + jb) * lda + j) * sizeof(float), lda, stream1));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the SSYRK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(float), stream0));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Perform the diagonal block decomposition using the CPU */
      spotrf(CBlasUpper, jb, B, ldb, info);
      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(float), stream0));
      /* Wait until the SGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Triangular solve of the diagonal block using the row matrix to the
       * right on the same stream as the copy to ensure it has completed first */
      CU_ERROR_CHECK(cuStrsm(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit,
                             jb, n - j - jb, one, A + (j * lda + j) * sizeof(float), lda,
                             A + ((j + jb) * lda + j) * sizeof(float), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                             -one, A + j * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));
      /* Overlap the SSYRK with an SGEMM (on a different stream) */
      CU_ERROR_CHECK(cuSgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                             -one, A + (j + jb) * sizeof(float), lda,
                             A + j * sizeof(float), lda,
                             one, A + (j * lda + j + jb) * sizeof(float), lda, stream1));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the SSYRK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(float), stream0));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Perform the diagonal block decomposition using the CPU */
      spotrf(CBlasLower, jb, B, ldb, info);
      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(float), stream0));
      /* Wait until the SGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Triangular solve of the diagonal block using the column matrix below
       * on the same stream as the copy to ensure it has completed first */
      CU_ERROR_CHECK(cuStrsm(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit,
                             n - j - jb, jb, one, A + (j * lda + j) * sizeof(float), lda,
                             A + (j * lda + j + jb) * sizeof(float), lda, stream0));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return (*info == 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMultiGPUSpotrf(CUmultiGPULAPACKhandle handle, CBlasUplo uplo,
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

  const size_t nb = (uplo == CBlasUpper) ? SGEMM_T_MB : SGEMM_N_NB;

  if (n < nb) {
    spotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUSsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                                     -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle->blas_handle));
      spotrf(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle->blas_handle, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j,
                                       -one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
                                       one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb,
                                       one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUSsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                                     -one, &A[j], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle->blas_handle));
      spotrf(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                                       -one, &A[j + jb], lda, &A[j], lda,
                                       one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb,
                                       one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
