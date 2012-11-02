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

static inline void dpotf2(CBlasUplo uplo, size_t n, double * restrict A, size_t lda, long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < n; i++) {
      register double temp = zero;
      const double * restrict B = A;
      for (size_t k = 0; k < i; k++)
        temp += A[i * lda + k] * B[i * lda + k];

      register double aii = A[i * lda + i] - temp;
      if (aii <= zero || isnan(aii)) {
        A[i * lda + i] = aii;
        *info = (long)i + 1;
        return;
      }
      aii = sqrt(aii);
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
        register double temp = A[k * lda + j];
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      register double ajj = A[j * lda + j];
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

void dpotrf(CBlasUplo uplo, size_t n, double * restrict A, size_t lda, long * restrict info) {
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
    dpotf2(uplo, n, A, lda, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      dsyrk(CBlasUpper, CBlasTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      dpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        dgemm(CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda);
        dtrsm(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      dsyrk(CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda);
      dpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        dgemm(CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda);
        dtrsm(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

/*static inline */CUresult cuDpotf2(CUmodule module, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, CUdeviceptr info, CUstream stream) {
  const unsigned int bx = 32;

  char name[39];
  snprintf(name, 39, "_Z6dpotf2IL9CBlasUplo%dELj%uEEviPdiPi", uplo, bx);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &A, &lda, &info };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuDpotrf(CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0) return CUDA_SUCCESS;

  double * B;
  size_t ldb;
  CUdeviceptr dInfo;
  CUstream stream0, stream1;
  CUmodule /*dpotf2,*/ dsyrk, dgemm, dtrsm;

  const size_t nb = (uplo == CBlasUpper) ? 192 : 576;

  CU_ERROR_CHECK(cuMemAlloc(&dInfo, sizeof(long)));
  CU_ERROR_CHECK(cuMemcpyHtoD(dInfo, info, sizeof(long)));

//   if (n < nb) {
//     CU_ERROR_CHECK(cuModuleLoad(&dpotf2, "dpotrf.cubin"));
//     CU_ERROR_CHECK(cuDpotf2(dpotf2, uplo, n, A, lda, dInfo, NULL));
//     CU_ERROR_CHECK(cuModuleUnload(dpotf2));
//     return CUDA_SUCCESS;
//   }

  CU_ERROR_CHECK(cuMemAllocHost((void **)&B, (ldb = (nb + 1u) & ~1u) * nb * sizeof(double)));

  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  CU_ERROR_CHECK(cuModuleLoad(&dsyrk, "dsyrk.cubin"));
//   CU_ERROR_CHECK(cuModuleLoad(&dpotf2, "dpotrf.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&dgemm, "dgemm.cubin"));
  CU_ERROR_CHECK(cuModuleLoad(&dtrsm, "dtrsm.cubin"));

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuDsyrk(dsyrk, CBlasUpper, CBlasTrans, jb, j, -one, A + j * lda * sizeof(double), lda, one, A + (j * lda + j) * sizeof(double), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double), stream0));
      CU_ERROR_CHECK(cuDgemm(dgemm, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, A + j * lda * sizeof(double), lda, A + (j + jb) * lda * sizeof(double), lda, one, A + ((j + jb) * lda + j) * sizeof(double), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      dpotrf(CBlasUpper, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuDtrsm(dtrsm, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, A + (j * lda + j) * sizeof(double), lda, A + ((j + jb) * lda + j) * sizeof(double), lda, stream0));
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuDsyrk(dsyrk, CBlasLower, CBlasNoTrans, jb, j, -one, A + j * sizeof(double), lda, one, A + (j * lda + j) * sizeof(double), lda, stream0));
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(double), stream0));
      CU_ERROR_CHECK(cuDgemm(dgemm, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, A + (j + jb) * sizeof(double), lda, A + j * sizeof(double), lda, one, A + (j * lda + j + jb) * sizeof(double), lda, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));
      dpotrf(CBlasLower, jb, B, ldb, info);
      if (*info != 0) {
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(double), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      CU_ERROR_CHECK(cuDtrsm(dtrsm, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, A + (j * lda + j) * sizeof(double), lda, A + (j * lda + j + jb) * sizeof(double), lda, stream0));
    }
  }

  CU_ERROR_CHECK(cuModuleUnload(dsyrk));
//   CU_ERROR_CHECK(cuModuleUnload(dpotf2));
  CU_ERROR_CHECK(cuModuleUnload(dgemm));
  CU_ERROR_CHECK(cuModuleUnload(dtrsm));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  CU_ERROR_CHECK(cuMemFreeHost(B));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDpotrf(CUcontext * contexts, int deviceCount, CBlasUplo uplo, size_t n, double * restrict A, size_t lda, long * restrict info) {
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
    dpotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUDsyrk(contexts, deviceCount, CBlasUpper, CBlasTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      dpotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUDtrsm(contexts, deviceCount, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUDsyrk(contexts, deviceCount, CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda));
      dpotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUDtrsm(contexts, deviceCount, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
