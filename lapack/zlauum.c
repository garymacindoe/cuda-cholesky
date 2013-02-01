#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include "config.h"
#include "zlauum.fatbin.c"

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

static const double complex zero = 0.0 + 0.0 * I;
static const double complex one = 1.0 + 0.0 * I;

static inline void zlauu2(CBlasUplo uplo,
                          size_t n,
                          double complex * restrict A, size_t lda) {
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register double complex ajj = conj(A[j * lda + j]);
      for (size_t i = 0; i <= j; i++)
        A[j * lda + i] *= ajj;

      for (size_t k = j + 1; k < n; k++) {
        register double complex temp = conj(A[k * lda + j]);
        for (size_t i = 0; i <= j; i++)
          A[j * lda + i] += temp * A[k * lda + i];
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        A[j * lda + i] *= conj(A[i * lda + i]);

        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += conj(A[i * lda + k]) * A[j * lda + k];
      }
    }
  }
}

void zlauum(CBlasUplo uplo,
            size_t n,
            double complex * restrict A, size_t lda,
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
    zlauu2(uplo, n, A, lda);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      ztrmm(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, i, ib,
            one, &A[i * lda + i], lda, &A[i * lda], lda);
      zlauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        zgemm(CBlasNoTrans, CBlasConjTrans, i, ib, n - i - ib,
              one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda], lda);
        zherk(CBlasUpper, CBlasNoTrans, ib, n - i - ib,
              one, &A[(i + ib) * lda + i], lda,
              one, &A[i * lda + i], lda);
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      ztrmm(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, ib, i,
            one, &A[i * lda + i], lda, &A[i], lda);
      zlauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        zgemm(CBlasConjTrans, CBlasNoTrans, ib, i, n - i - ib,
              one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
              one, &A[i], lda);
        zherk(CBlasLower, CBlasConjTrans, ib, n - i - ib,
              one, &A[i * lda + i + ib], lda,
              one, &A[i * lda + i], lda);
      }
    }
  }
}

static inline CUresult cuZlauu2(CUmodule module, CBlasUplo uplo,
                                size_t n,
                                CUdeviceptr A, size_t lda, CUstream stream) {
  const unsigned int bx = 16;

  char name[44];
  snprintf(name, 44, "_Z6zlauu2IL9CBlasUplo%dELj%uEEvP7double2ii", uplo, bx);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &A, &lda, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuZlauum(CUBLAShandle handle,
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

  double complex * B;
  CUdeviceptr X;
  size_t ldb, ldx;
  CUstream stream0, stream1;

  /**
   * In both loops for ZTRTRI and ZLAUUM A is updated column by column whether
   * upper or lower triangular.  The ZTRMM consumes most of the FLOPS in ZTRTRI
   * while the SGEMM consumes most of the FLOPS in ZLAUUM.  ZTRMM is always
   * called with transA == CBlasNoTrans as is SGEMM except in the lower
   * triangular ZLAUUM.  This means that the size of B in host memory changes
   * between loops when A is lower triangular.
   */

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    // Block size for upper triangular ZTRTRI and ZLAUUM
    const size_t nb = ZGEMM_N_MB;

    // Allocate page-locked host memory for diagonal block
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 1u) & ~1u) * sizeof(double complex)));

    // Allocate temporary column for out of place ZTRMM
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(double complex), nb, sizeof(double complex)));
    ldx /= sizeof(double complex);

    // Loop for ZLAUUM
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuZtrmm2(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, i, ib,
                              one, A + (i * lda + i) * sizeof(double complex), lda,
                              A + i * lda * sizeof(double complex), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuZgemm2(handle, CBlasNoTrans, CBlasConjTrans, i, ib, n - i - ib,
                              one, A + (i + ib) * lda * sizeof(double complex), lda,
                              A + ((i + ib) * lda + i) * sizeof(double complex), lda,
                              one, X, ldx, A + i * lda * sizeof(double complex), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host. */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                         ib, ib, sizeof(double complex), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the multiplication of the diagonal block using the CPU */
      zlauum(CBlasUpper, ib, B, ldb, info);
      /* Ensure the ZTRMM has finished before copying the block back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(double complex), stream1));
      /* Perform the ZHERK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuZherk(handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                             one, A + ((i + ib) * lda + i) * sizeof(double complex), lda,
                             one, A + (i * lda + i) * sizeof(double complex), lda, stream1));
      /* Ensure the ZHERK has finished before starting the ZTRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
    }
  }
  else {
    // Block size for lower triangular ZLAUUM
    const size_t mb = ZGEMM_CN_MB;

    // Allocate page-locked host memory for diagonal block
    CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (mb + 1u) & ~1u) * sizeof(double complex)));

    // Allocate temporary row for out of place ZTRMM in ZTRTRI
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, mb * sizeof(double complex), n, sizeof(double complex)));

    // Loop for ZLAUUM
    for (size_t i = 0; i < n; i += mb) {
      const size_t ib = min(mb, n - i);

      /* Update the current column using the diagonal block */
      CU_ERROR_CHECK(cuZtrmm2(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, ib, i,
                              one, A + (i * lda + i) * sizeof(double complex), lda,
                              A + i * sizeof(double complex), lda, X, ldx, stream0));
      /* Update the current column using the big matrix to the right */
      CU_ERROR_CHECK(cuZgemm2(handle, CBlasConjTrans, CBlasNoTrans, ib, i, n - i - ib,
                              one, A + (i * lda + i + ib) * sizeof(double complex), lda,
                              A + (i + ib) * sizeof(double complex), lda,
                              one, X, ldx, A + i * sizeof(double complex), lda, stream0));
      /* Overlap both the operations above with a copy of the diagonal block
       * onto the host. */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, i, i,
                                         ib, ib, sizeof(double complex), stream1));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Form the multiplication of the diagonal block using the CPU */
      zlauum(CBlasUpper, ib, B, ldb, info);
      /* Ensure the ZTRMM has finished before copying the block back */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, i, i, B, ldb, 0, 0,
                                         ib, ib, sizeof(double complex), stream1));
      /* Perform the ZHERK on the same stream as the copy to ensure A has
       * finised copying back first. */
      CU_ERROR_CHECK(cuZherk(handle, CBlasLower, CBlasConjTrans, ib, n - i - ib,
                             one, A + (i * lda + i + ib) * sizeof(double complex), lda,
                             one, A + (i * lda + i) * sizeof(double complex), lda, stream1));
      /* Ensure the ZHERK has finished before starting the ZTRMM from the next
       * iteration. (Only needed for devices that can execute multiple kernels
       * simultaneously.) */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));
  CU_ERROR_CHECK(cuMemFree(X));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZlauum(CUmultiGPUBLAShandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          double complex * restrict A, size_t lda,
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
    const size_t nb = ZGEMM_N_MB;

    // Upper triangular ZLAUUM
    for (size_t i = 0; i < n; i += nb) {
      const size_t ib = min(nb, n - i);

      CU_ERROR_CHECK(cuMultiGPUZtrmm(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit,
                                     i, ib, one, &A[i * lda + i], lda, &A[i * lda], lda));
      zlauu2(CBlasUpper, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasConjTrans, i, ib, n - i - ib,
                                       one, &A[(i + ib) * lda], lda, &A[(i + ib) * lda + i], lda,
                                       one, &A[i * lda], lda));
        CU_ERROR_CHECK(cuMultiGPUZherk(handle, CBlasUpper, CBlasNoTrans, ib, n - i - ib,
                                       one, &A[(i + ib) * lda + i], lda, one, &A[i * lda + i], lda));
      }
    }
  }
  else {
    const size_t mb = ZGEMM_CN_MB;

    // Lower triangular ZLAUUM
    for (size_t i = 0; i < n; i += mb) {
      const size_t ib = min(mb, n - i);

      CU_ERROR_CHECK(cuMultiGPUZtrmm(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, ib, i,
                                     one, &A[i * lda + i], lda, &A[i], lda));
      zlauu2(CBlasLower, ib, &A[i * lda + i], lda);

      if (i + ib < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasConjTrans, CBlasNoTrans, ib, i, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, &A[i + ib], lda,
                                       one, &A[i], lda));
        CU_ERROR_CHECK(cuMultiGPUZherk(handle, CBlasLower, CBlasConjTrans, ib, n - i - ib,
                                       one, &A[i * lda + i + ib], lda, one, &A[i * lda + i], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
