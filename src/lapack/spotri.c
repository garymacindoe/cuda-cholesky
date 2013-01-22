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

static void slauum(CBlasUplo uplo,
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
              one, &A[i], lda,
              one, &A[i * lda + i], lda);
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
  if (*info != 0)
    return;
  slauum(uplo, n, A, lda, info);
}

CUresult cuSpotri(CBlasUplo uplo,
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
  CUdeviceptr X;
  size_t ldb, ldx;
  CUmodule sgemm, ssyrk, strmm, strsm;
  CUstream stream0, stream1;

  /**
   * In both loops for STRTRI and SLAUUM A is updated column by column whether
   * upper or lower triangular.  The STRMM consumes most of the FLOPS in STRTRI
   * while the SGEMM consumes most of the FLOPS in SLAUUM.  STRMM is always
   * called with transA == CBlasNoTrans as is SGEMM except in the lower
   * triangular SLAUUM.  This means that the size of B in host memory changes
   * between loops when A is lower triangular.
   */

  // Load the GPU BLAS modules
  CU_ERROR_CHECK(cuModuleLoad(&sgemm, "sgemm.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&ssyrk, "ssyrk.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&strmm, "strmm.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&strsm, "strsm.fatbin"));

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
      CU_ERROR_CHECK(cuStrmm2(strmm, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, j, jb,
                              one, A, lda, A + j * lda * sizeof(float), lda, X, ldx, stream0));
      /* GPU STRMM is out of place so copy back into place */
      CU_ERROR_CHECK(cuMemcpyDtoD2DAsync(A, lda, 0, j, X, ldx, 0, 0, j, jb, sizeof(float), stream0));
      /* Then update the column again using the small square matrix on the
       * diagonal below (on the same stream) */
      CU_ERROR_CHECK(cuStrsm(strsm, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, j, jb,
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
      strtri(CBlasUpper, CBlasNonUnit, jb, B, ldb, info);
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

    // Loop for SLAUUM
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm2(strmm, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, i, ib,
                              one, A + (i * lda + i) * sizeof(float), lda,
                              A + i * lda * sizeof(float), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm2(sgemm, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                              one, A + (i + ib) * lda * sizeof(float), lda,
                              A + ((i + ib) * lda + i) * sizeof(float), lda,
                              one, X, ldx, A + i * lda * sizeof(float), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host. */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                         ib, ib, sizeof(float), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the multiplication of the diagonal block using the CPU */
      slauum(CBlasUpper, ib, B, ldb, info);
      /* Ensure the STRMM has finished before copying the block back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(float), stream1));
      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(ssyrk, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                             one, A + ((i + ib) * lda + i) * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream1));
      /* Ensure the SSYRK has finished before starting the STRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
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
      CU_ERROR_CHECK(cuStrmm2(strmm, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, n - j - jb, jb,
                              one, A + ((j + jb) * lda + j + jb) * sizeof(float), lda,
                              A + (j * lda + j + jb) * sizeof(float), lda, X, ldx, stream0));
      /* GPU STRMM is out of place so copy back into place */
      CU_ERROR_CHECK(cuMemcpyDtoD2DAsync(A, lda, j + jb, j, X, ldx, 0, 0, j, jb, sizeof(float), stream0));
      /* Then update the column again using the small square matrix on the
       * diagonal above (on the same stream) */
      CU_ERROR_CHECK(cuStrsm(strsm, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, n - j - jb, jb,
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
      strtri(CBlasLower, CBlasNonUnit, jb, B, ldb, info);
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

    // Block size for lower triangular SLAUUM
    const size_t mb = SGEMM_T_MB;

    // Reallocate diagonal block
    CU_ERROR_CHECK(cuMemFreeHost(B));
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (mb + 3u) & ~3u) * sizeof(float)));

    // Reallocate temporary column as temporary row
    CU_ERROR_CHECK(cuMemFree(X));
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, mb * sizeof(float), n, sizeof(float)));

    // Loop for SLAUUM
    for (size_t i = 0; i < n; i += mb) {
      const size_t ib = min(mb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuStrmm2(strmm, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                              one, A + (i * lda + i) * sizeof(float), lda,
                              A + i * sizeof(float), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuSgemm2(sgemm, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                              one, A + (i * lda + i + ib) * sizeof(float), lda,
                              A + (i + ib) * sizeof(float), lda,
                              one, X, ldx, A + i * sizeof(float), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host. */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                         ib, ib, sizeof(float), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the multiplication of the diagonal block using the CPU */
      slauum(CBlasUpper, ib, B, ldb, info);
      /* Ensure the STRMM has finished before copying the block back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(float), stream1));
      /* Perform the SSYRK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuSsyrk(ssyrk, CBlasLower, CBlasTrans, ib, n - i - ib,
                             one, A + i * sizeof(float), lda,
                             one, A + (i * lda + i) * sizeof(float), lda, stream1));
      /* Ensure the SSYRK has finished before starting the STRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));
  CU_ERROR_CHECK(cuMemFree(X));

  CU_ERROR_CHECK(cuModuleUnload(sgemm));
  CU_ERROR_CHECK(cuModuleUnload(ssyrk));
  CU_ERROR_CHECK(cuModuleUnload(strmm));
  CU_ERROR_CHECK(cuModuleUnload(strsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return (*info == 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMultiGPUSpotri(CUmultiGPUBlasHandle handle,
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

    // Upper triangular STRTRI
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);
      CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit,
                                     j, jb, one, A, lda, &A[j * lda], lda));
      CU_ERROR_CHECK(cuMultiGPUStrsm(handle, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit,
                                     j, jb, -one, &A[j * lda + j], lda, &A[j * lda], lda));
      strtri(CBlasUpper, CBlasNonUnit, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
    }

    // Upper triangular SLAUUM
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit,
                                     i, ib, one, &A[i * lda + i], lda, &A[i * lda], lda));
      slauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle, CBlasNoTrans, CBlasTrans, i, ib, n - i - ib,
                                       one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
                                       one, &A[i * lda], lda));
        CU_ERROR_CHECK(cuMultiGPUSsyrk(handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                                       one, &A[(i + ib) * lda + i], lda, one, &A[i * lda + i], lda));
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
        CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit,
                                       n - j - jb, jb,
                                       one, &A[(j + jb) * lda + j + jb], lda,
                                       &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handle, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit,
                                       n - j - jb, jb,
                                       -one, &A[j * lda + j], lda,
                                       &A[j * lda + j + jb], lda));
      }
      strtri(CBlasLower, CBlasNonUnit, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
    } while (j > 0);

    // Lower triangular SLAUUM
    const size_t mb = SGEMM_T_MB;
    for (size_t i = 0; i < n; i += mb) {
      const size_t ib = min(mb, n - i);

      CU_ERROR_CHECK(cuMultiGPUStrmm(handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, ib, i,
                                     one, &A[i * lda + i], lda, &A[i], lda));
      slauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handle, CBlasTrans, CBlasNoTrans, ib, i, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
                                       one, &A[i], lda));
        CU_ERROR_CHECK(cuMultiGPUSsyrk(handle, CBlasLower, CBlasTrans, ib, n - i - ib,
                                       one, &A[i], lda, one, &A[i * lda + i], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
