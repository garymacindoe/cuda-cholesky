#include "lapack.h"
#include "error.h"
#include <stdio.h>

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

static inline void zpotf2(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register double aii = creal(A[i * lda + i]);
      for (size_t k = 0; k < i; k++)
        aii -= A[i * lda + k] * conj(A[i * lda + k]);
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrt(aii);
      A[i * lda + i] = aii;

      for (size_t j = i + 1; j < n; j++) {
        register double complex temp = A[j * lda + i];;
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * conj(A[i * lda + k]);
        A[j * lda + i] = temp / aii;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      register double ajj = creal(A[j * lda + j]);
      for (size_t k = 0; k < j; k++)
        ajj -= A[k * lda + j] * conj(A[k * lda + j]);
      if (ajj <= zero || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j + 1;
        return;
      }
      ajj = sqrt(ajj);
      A[j * lda + j] = ajj;

      for (size_t k = 0; k < j; k++) {
        register double complex temp = conj(A[k * lda + j]);
        for (size_t i = j + 1; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

void zpotrf(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
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

      zherk(CBlasUpper, CBlasConjTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda);
        ztrsm(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      zherk(CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda);
      zpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        zgemm(CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda);
        ztrsm(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

static inline CUresult cuZpotf2(CUmodule module, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, CUdeviceptr info, CUstream stream) {
  const unsigned int bx = (uplo == CBlasUpper) ?  8 : 16;
  const unsigned int by = (uplo == CBlasUpper) ?  8 :  4;

  char name[43];
  snprintf(name, 43, "_Z6zpotf2IL9CBlasUplo%dELj%uELj%uEEviPfiPi", uplo, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &A, &lda, &info };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuZpotrf(CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  double complex * B;
  size_t ldb;
  CUdeviceptr dInfo;
  CUstream stream0, stream1;
  CUmodule /*zpotf2,*/ zherk, zgemm, ztrsm;

  const size_t nb = (uplo == CBlasUpper) ? 192 : 576;

  CU_ERROR_CHECK(cuMemAlloc(&dInfo, sizeof(long)));
  CU_ERROR_CHECK(cuMemcpyHtoD(dInfo, info, sizeof(long)));

//   if (n < nb) {
//     CU_ERROR_CHECK(cuModuleLoad(&zpotf2, "zpotrf.cubin"));
//     CU_ERROR_CHECK(cuZpotf2(zpotf2, uplo, n, A, lda, dInfo, NULL));
//     CU_ERROR_CHECK(cuModuleUnload(zpotf2));
//     return CUDA_SUCCESS;
//   }

  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = nb) * nb * sizeof(double complex)));

  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  CU_ERROR_CHECK(cuModuleLoad(&zherk, "zherk.cubin"));
//   CU_ERROR_CHECK(cuModuleLoad(&zpotf2, "zpotrf.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&zgemm, "zgemm.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&ztrsm, "ztrsm.cubin"));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuZherk(zherk, CBlasUpper, CBlasConjTrans, jb, j, -one, A + j * lda * sizeof(double complex), lda, one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, A + j * lda * sizeof(double complex), lda, A + (j + jb) * lda * sizeof(double complex), lda, complex_one, A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      zpotrf(CBlasUpper, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuZtrsm(ztrsm, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda, A + ((j + jb) * lda + j) * sizeof(double complex), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuZherk(zherk, CBlasLower, CBlasNoTrans, jb, j, -one, A + j * sizeof(double complex), lda, one, A + (j * lda + j) * sizeof(double complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuZgemm(zgemm, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, A + (j + jb) * sizeof(double complex), lda, A + j * sizeof(double complex), lda, complex_one, A + (j * lda + j + jb) * sizeof(double complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      zpotrf(CBlasLower, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuZtrsm(ztrsm, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, A + (j * lda + j) * sizeof(double complex), lda, A + (j * lda + j + jb) * sizeof(double complex), lda, stream0));
    }
  }

  CU_ERROR_CHECK(cuModuleUnload(zherk));
//   CU_ERROR_CHECK(cuModuleUnload(zpotf2));
  CU_ERROR_CHECK(cuModuleUnload(zgemm));
  CU_ERROR_CHECK(cuModuleUnload(ztrsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  CU_ERROR_CHECK(cuMemFreeHost(B));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZpotrf(CUcontext * contexts, int deviceCount, CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  const size_t nb = 1024;

  if (n < nb) {
    zpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(contexts, deviceCount, CBlasUpper, CBlasConjTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      zpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(contexts, deviceCount, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, &A[j * lda], lda, &A[(j + jb) * lda], lda, complex_one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUZtrsm(contexts, deviceCount, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUZherk(contexts, deviceCount, CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda));
      zpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUZgemm(contexts, deviceCount, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, &A[j + jb], lda, &A[j], lda, complex_one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUZtrsm(contexts, deviceCount, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
