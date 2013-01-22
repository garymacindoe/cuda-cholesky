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

static const double zero = 0.0;
static const double one = 1.0;
static const double complex complex_zero = 0.0 + 0.0 * I;
static const double complex complex_one = 1.0 + 0.0 * I;

static inline void zpotf2(CBlasUplo uplo,
                          size_t n,
                          double complex * restrict A, size_t lda,
                          long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register double temp = zero;
      const double complex * B = &A[i * lda];
      for (size_t k = 0; k < i; k++)
        temp += A[i * lda + k] * conj(B[k]);

      register double aii = creal(A[i * lda + i]) - temp;
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrt(aii);
      A[i * lda + i] = aii;

      for (size_t j = i + 1; j < n; j++) {
        register double complex temp = complex_zero;
        for (size_t k = 0; k < i; k++)
          temp += A[j * lda + k] * conj(A[i * lda + k]);
        A[j * lda + i] = (A[j * lda + i] - temp) / aii;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < j; k++) {
        register double complex temp = conj(A[k * lda + j]);
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      register double ajj = creal(A[j * lda + j]);
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

void zpotrf(CBlasUplo uplo,
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

  if (n < nb) {
    zpotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      zherk(CBlasUpper, CBlasConjTrans, jb, j,
            -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j,
              -one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
              one, &A[(j + jb) * lda + j], lda);
        ztrsm(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb,
              one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      zherk(CBlasLower, CBlasNoTrans, jb, j,
            -one, &A[j], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j,
              -one, &A[j + jb], lda, &A[j], lda,
              one, &A[j * lda + j + jb], lda);
        ztrsm(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb,
              one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

// static inline CUresult cuZpotf2(CUmodule module, CBlasUplo uplo,
//                                 size_t n,
//                                 CUdeviceptr A, size_t lda,
//                                 CUdeviceptr info, CUstream stream) {
//   const unsigned int bx = 32;
//
//   char name[45];
//   snprintf(name, 45, "_Z6zpotf2IL9CBlasUplo%dELj%uEEviP7double2iPi", uplo, bx);
//
//   CUfunction function;
//   CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));
//
//   void * params[] = { &n, &A, &lda, &info };
//
//   CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));
//
//   return CUDA_SUCCESS;
// }

CUresult cuZpotrf(CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
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
   * The ZGEMM consumes most of the FLOPs in the Cholesky decomposition so the
   * block sizes are chosen to favour it.  In the upper triangular case it is
   * the row matrix to the right of the diagonal block that is updated via
   * D = -A^T * C + D (i.e. the A argument to ZGEMM is transposed) therefore the
   * block size is ZGEMM_C_MB.  For the lower triangular case it is the column
   * matrix below the diagonal block that is updated via D = -C * A^T + D so the
   * block size is ZGEMM_N_NB.
   */
  const size_t nb = (uplo == CBlasUpper) ? ZGEMM_CN_MB : ZGEMM_N_NB;

  double complex * B;
  size_t ldb;
  CUmodule zherk, zgemm, ztrsm;
  CUstream stream0, stream1;

  // Allocate page-locked host memory for diagonal block
  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 1u) & ~1u) * sizeof(double complex)));

  // Load the GPU BLAS modules
  CU_ERROR_CHECK(cuModuleLoad(&zherk, "zherk.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&zgemm, "zgemm.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&ztrsm, "ztrsm.fatbin"));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuZherk(zherk, CBlasUpper, CBlasConjTrans, jb, j,
                             -one, A + j * lda * sizeof(double complex), lda,
                             one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      /* Overlap the ZHERK with a ZGEMM (on a different stream) */
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j,
                             -complex_one, A + j * lda * sizeof(double complex), lda,
                             A + (j + jb) * lda * sizeof(double complex), lda,
                             complex_one, A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream1));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the ZHERK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(double complex), stream0));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Perform the diagonal block decomposition using the CPU */
      zpotrf(CBlasUpper, jb, B, ldb, info);
      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(double complex), stream0));
      /* Wait until the ZGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Triangular solve of the diagonal block using the row matrix to the
       * right on the same stream as the copy to ensure it has completed first */
      CU_ERROR_CHECK(cuZtrsm(ztrsm, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit,
                             jb, n - j - jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda,
                             A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuZherk(zherk, CBlasLower, CBlasNoTrans, jb, j,
                             -one, A + j * sizeof(double complex), lda,
                             one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      /* Overlap the ZHERK with a ZGEMM (on a different stream) */
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j,
                             -complex_one, A + (j + jb) * sizeof(double complex), lda,
                             A + j * sizeof(double complex), lda,
                             complex_one, A + (j * lda + j + jb) * sizeof(double complex), lda, stream1));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the ZHERK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j,
                                         jb, jb, sizeof(double complex), stream0));
      /* Wait until the diagonal block has been copied */
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      /* Perform the diagonal block decomposition using the CPU */
      zpotrf(CBlasLower, jb, B, ldb, info);
      /* Check for positive definite matrix */
      if (*info != 0) {
        *info += (long)j;
        break;
      }
      /* Copy the diagonal block back onto the device */
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                         jb, jb, sizeof(double complex), stream0));
      /* Wait until the ZGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
//       CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Triangular solve of the diagonal block using the column matrix below
       * on the same stream as the copy to ensure it has completed first */
      CU_ERROR_CHECK(cuZtrsm(ztrsm, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit,
                             n - j - jb, jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda,
                             A + (j * lda + j + jb) * sizeof(double complex), lda, stream0));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(B));

  CU_ERROR_CHECK(cuModuleUnload(zherk));
  CU_ERROR_CHECK(cuModuleUnload(zgemm));
  CU_ERROR_CHECK(cuModuleUnload(ztrsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return (*info == 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMultiGPUZpotrf(CUmultiGPUBlasHandle handle, CBlasUplo uplo,
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

  const size_t nb = (uplo == CBlasUpper) ? ZGEMM_CN_MB : ZGEMM_N_NB;

  if (n < nb) {
    zpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(handle, CBlasUpper, CBlasConjTrans, jb, j,
                                     -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
      zpotrf(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j,
                                       -complex_one, &A[j * lda], lda, &A[(j + jb) * lda], lda,
                                       complex_one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUZtrsm(handle, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb,
                                       complex_one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(handle, CBlasLower, CBlasNoTrans, jb, j,
                                     -one, &A[j], lda, one, &A[j * lda + j], lda));
      CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
      zpotrf(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j,
                                       -complex_one, &A[j + jb], lda, &A[j], lda,
                                       complex_one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUZtrsm(handle, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb,
                                       complex_one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
