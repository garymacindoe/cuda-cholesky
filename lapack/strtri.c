#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include "config.h"
#include "strtri.fatbin.c"

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

      size_t i = n - 1;
      do {
        register float temp = A[j * lda + i];
        if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + k] += temp * A[i * lda + k];
      } while (i-- > j);
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] *= ajj;
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
      size_t i = n - 1;
      do {
        register float temp = B[j * ldb + i];
        if (diag == CBlasNonUnit) B[j * ldb + i] *= B[i * ldb + i];
        for (size_t k = i + 1; k < n; k++)
          B[j * ldb + k] += temp * B[i * ldb + k];
      } while (i-- > j);
      for (size_t i = j + 1; i < n; i++)
        B[j * ldb + i] *= bjj;
    } while (j-- > 0);
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
             &B[j * ldb], lda);
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

CUresult cuStrtri(CUblashandle handle,
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

  float * B;
  CUdeviceptr X;
  size_t ldb, ldx;
  CUstream stream0, stream1;

  /**
   * In both loops for STRTRI and SLAUUM A is updated column by column whether
   * upper or lower triangular.  The STRMM consumes most of the FLOPS in STRTRI
   * while the SGEMM consumes most of the FLOPS in SLAUUM.  STRMM is always
   * called with transA == CBlasNoTrans as is SGEMM except in the lower
   * triangular SLAUUM.  This means that the size of B in host memory changes
   * between loops when A is lower triangular.
   */

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    // Block size for upper triangular STRTRI and SLAUUM
    const size_t nb = SGEMM_N_MB;

    // Allocate page-locked host memory for diagonal block
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 3u) & ~3u) * sizeof(float)));

    // Allocate temporary column for out of place STRMM
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(float), nb, sizeof(float)));
    ldx /= sizeof(float);

    // Loop for STRTRI
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Update the current column using the big square matrix to the left */
      CU_ERROR_CHECK(cuStrmm2(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, diag, j, jb,
                              one, A, lda, A + j * lda * sizeof(float), lda, X, ldx, stream0));
      /* GPU STRMM is out of place so copy back into place */
      CU_ERROR_CHECK(cuMemcpyDtoD2DAsync(A, lda, 0, j, X, ldx, 0, 0, j, jb, sizeof(float), stream0));
      /* Then update the column again using the small square matrix on the
       * diagonal below (on the same stream) */
      CU_ERROR_CHECK(cuStrsm(handle, CBlasRight, CBlasUpper, CBlasNoTrans, diag, j, jb,
                             -one, A + (j * lda + j) * sizeof(float), lda, A + j * lda * sizeof(float), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host.  There is a possibility of overwriting the result of the
       * previous iteration's block inverse in host memory before it has been
       * copied back to the GPU if the GPU can do more than one copy at once (CC
       * 2.x). */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(float), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the inverse of the diagonal block using the CPU */
      strtri(CBlasUpper, diag, jb, B, ldb, info);
      /* Check for singular matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device using the same stream as
       * the STRSM to ensure it is finished reading the diagonal block before
       * the new one is copied */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(float), stream0));
    }
  }
  else {
    // Block size for upper triangular STRTRI
    const size_t nb = SGEMM_N_MB;

    // Allocate page-locked host memory for diagonal block
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 3u) & ~3u) * sizeof(float)));

    // Allocate temporary column for out of place STRMM in STRTRI
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(float), nb, sizeof(float)));
    ldx /= sizeof(float);

    // Loop for STRTRI
    const size_t r = n % nb;
    size_t j = (r == 0) ? n : n + nb - r;
    do {
      j -= nb;

      const size_t jb = min(nb, n - j);

      /* Update the current column using the big square matrix to the right */
      CU_ERROR_CHECK(cuStrmm2(handle, CBlasLeft, CBlasLower, CBlasNoTrans, diag, n - j - jb, jb,
                              one, A + ((j + jb) * lda + j + jb) * sizeof(float), lda,
                              A + (j * lda + j + jb) * sizeof(float), lda, X, ldx, stream0));
      /* GPU STRMM is out of place so copy back into place */
      CU_ERROR_CHECK(cuMemcpyDtoD2DAsync(A, lda, j + jb, j, X, ldx, 0, 0, j, jb, sizeof(float), stream0));
      /* Then update the column again using the small square matrix on the
       * diagonal above (on the same stream) */
      CU_ERROR_CHECK(cuStrsm(handle, CBlasRight, CBlasLower, CBlasNoTrans, diag, n - j - jb, jb,
                             -one, A + (j * lda + j) * sizeof(float), lda, A + (j * lda + j + jb) * sizeof(float), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host.  There is a possibility of overwriting the result of the
       * previous iteration's block inverse in host memory before it has been
       * copied back to the GPU if the GPU can do more than one copy at once (CC
       * 2.x). */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(float), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the inverse of the diagonal block using the CPU */
      strtri(CBlasLower, diag, jb, B, ldb, info);
      /* Check for singular matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device using the same stream as
       * the STRSM to ensure it is finished reading the diagonal block before
       * the new one is copied */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(float), stream0));
    } while (j > 0);
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));
  CU_ERROR_CHECK(cuMemFree(X));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUStrtri(CUmultiGPUBlasHandle handle,
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
      CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, diag,
                                     j, jb, one, A, lda, &A[j * lda], lda));
      CU_ERROR_CHECK(cuMultiGPUStrsm(handle, CBlasRight, CBlasUpper, CBlasNoTrans, diag,
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
        CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasLeft, CBlasLower, CBlasNoTrans, diag,
                                       n - j - jb, jb,
                                       one, &A[(j + jb) * lda + j + jb], lda,
                                       &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handle, CBlasRight, CBlasLower, CBlasNoTrans, diag,
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

