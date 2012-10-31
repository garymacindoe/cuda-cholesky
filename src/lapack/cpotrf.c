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

static const float zero = 0.0f;
static const float one = 1.0f;
static const float complex complex_zero = 0.0f + 0.0f * I;
static const float complex complex_one = 1.0f + 0.0f * I;

static inline void cpotf2(CBlasUplo uplo, size_t n, float complex * restrict A, size_t lda, long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register float aii = crealf(A[i * lda + i]);
      for (size_t k = 0; k < i; k++)
        aii -= A[i * lda + k] * conjf(A[i * lda + k]);
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrtf(aii);
      A[i * lda + i] = aii;

      for (size_t j = i + 1; j < n; j++) {
        register float complex temp = A[j * lda + i];;
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * conjf(A[i * lda + k]);
        A[j * lda + i] = temp / aii;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      register float ajj = crealf(A[j * lda + j]);
      for (size_t k = 0; k < j; k++)
        ajj -= A[k * lda + j] * conjf(A[k * lda + j]);
      if (ajj <= zero || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j + 1;
        return;
      }
      ajj = sqrtf(ajj);
      A[j * lda + j] = ajj;

      for (size_t k = 0; k < j; k++) {
        register float complex temp = conjf(A[k * lda + j]);
        for (size_t i = j + 1; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

void cpotrf(CBlasUplo uplo, size_t n, float complex * restrict A, size_t lda, long * restrict info) {
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
    cpotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      cherk(CBlasUpper, CBlasConjTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      cpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        cgemm(CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda);
        ctrsm(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      cherk(CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda);
      cpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        cgemm(CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda);
        ctrsm(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

static inline CUresult cuCpotf2(CUmodule module, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, CUdeviceptr info, CUstream stream) {
  const unsigned int bx = (uplo == CBlasUpper) ?  8 : 16;
  const unsigned int by = (uplo == CBlasUpper) ?  8 :  4;

  char name[43];
  snprintf(name, 43, "_Z6cpotf2IL9CBlasUplo%dELj%uELj%uEEviPfiPi", uplo, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &A, &lda, &info };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuCpotrf(CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  float complex * B;
  size_t ldb;
  CUdeviceptr dInfo;
  CUstream stream0, stream1;
  CUmodule /*cpotf2,*/ cherk, cgemm, ctrsm;

  const size_t nb = (uplo == CBlasUpper) ? 192 : 576;

  CU_ERROR_CHECK(cuMemAlloc(&dInfo, sizeof(long)));
  CU_ERROR_CHECK(cuMemcpyHtoD(dInfo, info, sizeof(long)));

//   if (n < nb) {
//     CU_ERROR_CHECK(cuModuleLoad(&cpotf2, "cpotrf.cubin"));
//     CU_ERROR_CHECK(cuSpotf2(cpotf2, uplo, n, A, lda, dInfo, NULL));
//     CU_ERROR_CHECK(cuModuleUnload(cpotf2));
//     return CUDA_SUCCESS;
//   }

  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 1u) & ~1u) * nb * sizeof(float complex)));

  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  CU_ERROR_CHECK(cuModuleLoad(&cherk, "cherk.cubin"));
//   CU_ERROR_CHECK(cuModuleLoad(&cpotf2, "cpotrf.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&cgemm, "cgemm.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&ctrsm, "ctrsm.cubin"));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuCherk(cherk, CBlasUpper, CBlasConjTrans, jb, j, -one, A + j * lda * sizeof(float complex), lda, one, A + (j * lda + j) * sizeof(float complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(float complex), stream0));
      CU_ERROR_CHECK(cuCgemm(cgemm, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, A + j * lda * sizeof(float complex), lda, A + (j + jb) * lda * sizeof(float complex), lda, complex_one, A + ((j + jb) * lda + j) * sizeof(float complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      cpotrf(CBlasUpper, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(float complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuCtrsm(ctrsm, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, A + (j * lda + j) * sizeof(float complex), lda, A + ((j + jb) * lda + j) * sizeof(float complex), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuCherk(cherk, CBlasLower, CBlasNoTrans, jb, j, -one, A + j * sizeof(float complex), lda, one, A + (j * lda + j) * sizeof(float complex), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(float complex), stream0));
      CU_ERROR_CHECK(cuCgemm(cgemm, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, A + (j + jb) * sizeof(float complex), lda, A + j * sizeof(float complex), lda, complex_one, A + (j * lda + j + jb) * sizeof(float complex), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      cpotrf(CBlasLower, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(float complex), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuCtrsm(ctrsm, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, A + (j * lda + j) * sizeof(float complex), lda, A + (j * lda + j + jb) * sizeof(float complex), lda, stream0));
    }
  }

  CU_ERROR_CHECK(cuModuleUnload(cherk));
//   CU_ERROR_CHECK(cuModuleUnload(cpotf2));
  CU_ERROR_CHECK(cuModuleUnload(cgemm));
  CU_ERROR_CHECK(cuModuleUnload(ctrsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  CU_ERROR_CHECK(cuMemFreeHost(B));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUCpotrf(CUcontext * contexts, int deviceCount, CBlasUplo uplo, size_t n, float complex * restrict A, size_t lda, long * restrict info) {
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
    cpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUCherk(contexts, deviceCount, CBlasUpper, CBlasConjTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      cpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasConjTrans, CBlasNoTrans, jb, n - j - jb, j, -complex_one, &A[j * lda], lda, &A[(j + jb) * lda], lda, complex_one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUCtrsm(contexts, deviceCount, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, jb, n - j - jb, complex_one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUCherk(contexts, deviceCount, CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda));
      cpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, CBlasConjTrans, n - j - jb, jb, j, -complex_one, &A[j + jb], lda, &A[j], lda, complex_one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUCtrsm(contexts, deviceCount, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, n - j - jb, jb, complex_one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
