#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include "../handle.h"

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

static const float bandwidth_htod = 5724.59f;   // MB/s
static const float overhead_htod = 0.009f;      // ms

static inline float time_nbnb_htod(size_t nb) {
  return (float)nb * ((float)(nb * sizeof(float)) / (bandwidth_htod * 1048.576f) + overhead_htod);
}

static inline float time_nnb_htod(size_t n, size_t nb) {
  return (float)(n * nb * sizeof(float)) / (bandwidth_htod * 1048.576f) + overhead_htod;
}

static const float bandwidth_dtoh = 5596.15f;   // MB/s
static const float overhead_dtoh = 0.106f;      // ms

static inline float time_nbnb_dtoh(size_t nb) {
  return (float)nb * ((float)(nb * sizeof(float)) / (bandwidth_dtoh * 1048.576f) + overhead_dtoh);
}

static inline float time_nnb_dtoh(size_t n, size_t nb) {
  return (float)(n * nb * sizeof(float)) / (bandwidth_dtoh * 1048.576f) + overhead_dtoh;
}

/**
 * Returns true if the time taken to transfer a block/column onto the host,
 * factorise and transfer back is too long to be hidden by the GPU SGEMM using
 * the particular block size.
 */
// static inline bool stay_on_device(CBlasUplo uplo, size_t n, size_t nb) {
//   return false;
// }

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

      ssyrk(CBlasUpper, CBlasTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda);
      spotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        sgemm(CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda);
        strsm(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda);
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      ssyrk(CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda);
      spotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return;
      }

      if (j + jb < n) {
        sgemm(CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda);
        strsm(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda);
      }
    }
  }
}

static inline CUresult cuSpotf2(CUhandle handle, CBlasUplo uplo,
                                size_t n,
                                CUdeviceptr A, size_t lda,
                                CUdeviceptr info, CUstream stream) {
  const unsigned int bx = 64;

  char name[39];
  snprintf(name, 39, "_Z6spotf2IL9CBlasUplo%dELj%uEEviPfiPi", uplo, bx);

  CUmodule module;
  CU_ERROR_CHECK(cuHandleGetModule(handle, &module, CU_HANDLE_SINGLE, CU_HANDLE_POTRF));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &A, &lda, &info };

  CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, bx, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuSpotrf(CUhandle handle, CBlasUplo uplo, size_t n, CUdeviceptr A, size_t lda, long * info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0)
    return CUDA_SUCCESS;

  float * B, * Binv;
  CUdeviceptr dInfo, dBinv, Dtemp;
  size_t ldb, dldb, dldd;
  CUstream stream0, stream1;

  const size_t nb = (uplo == CBlasUpper) ? 192 : 576;

  CU_ERROR_CHECK(cuHandleMemAlloc(handle, &dInfo, sizeof(long)));
  CU_ERROR_CHECK(cuMemcpyHtoD(dInfo, info, sizeof(long)));

  if (64 > n) {
    CU_ERROR_CHECK(cuSpotf2(handle, uplo, n, A, lda, dInfo, NULL));
    CU_ERROR_CHECK(cuMemcpyDtoH(info, dInfo, sizeof(long)));
    return CUDA_SUCCESS;
  }

  CU_ERROR_CHECK(cuHandleMemAllocPitchHost(handle, (void **)&B, &ldb, nb * sizeof(float), nb * 2, sizeof(float)));
  ldb /= sizeof(float);
  Binv = &B[nb * ldb];

  CU_ERROR_CHECK(cuHandleGetStream(handle, &stream0, 0));
  CU_ERROR_CHECK(cuHandleGetStream(handle, &stream1, 0));

  if (uplo == CBlasUpper) {
    CU_ERROR_CHECK(cuHandleMemAllocPitch(handle, &dBinv, &dldb, nb * sizeof(float), n, sizeof(float)));
    dldb /= sizeof(float);
    Dtemp = dBinv + nb * dldb * sizeof(float);
    dldd = dldb;

    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuSsyrk(handle, CBlasUpper, CBlasTrans, jb, j, -one, A + j * lda * sizeof(float), lda, one, A + (j * lda + j) * sizeof(float), lda, stream0));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the SSYRK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(float), stream0));
      /* Overlap the copy of the diagonal block with an out of place SGEMM (on a
       * different stream) which copies the row matrix to the right of the
       * diagonal into a temporary matrix Dtemp */
      CU_ERROR_CHECK(cuSgemm2(handle, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, A + j * lda * sizeof(float), lda, A + (j + jb) * lda * sizeof(float), lda, one, A + ((j + jb) * lda + j) * sizeof(float), lda, Dtemp, dldd, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));     // Synchronise to ensure diagonal block is copied
      spotrf(CBlasUpper, jb, B, ldb, info);             // before factorising on the host,
      if (*info != 0) {                                 // checking for positive definiteness
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      // Start copying the diagonal block back onto the device into the correct place
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(float), stream0));
      // and overlap that with an out-of-place inverse of the diagonal block via
      // the cholesky decomposition into Binv
      strtri2(CBlasUpper, CBlasNonUnit, jb, B, ldb, Binv, ldb, info);
      // Copy the inverse into Binv on the device
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(dBinv, dldb, 0, 0, Binv, ldb, 0, 0, jb, jb, sizeof(float), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));     // Synchronise to ensure SGEMM is finished
      // Start the out-of-place triangular matrix multiply to copy Dtemp back
      // into A using the inverse of the diagonal block.  This is done on the
      // same stream as the copy to ensure it has completed first.
      CU_ERROR_CHECK(cuStrmm2(handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, dBinv, dldb, Dtemp, dldd, A + ((j + jb) * lda + j) * sizeof(float), lda, stream0));
    }
  }
  else {
    CU_ERROR_CHECK(cuHandleMemAllocPitch(handle, &dBinv, &dldb, n * sizeof(float), nb, sizeof(float)));
    dldb /= sizeof(float);
    Dtemp = dBinv + nb * sizeof(float);
    dldd = dldb;

    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuSsyrk(handle, CBlasLower, CBlasNoTrans, jb, j, -one, A + j * sizeof(float), lda, one, A + (j * lda + j) * sizeof(float), lda, stream0));
      /* Start copying diagonal block onto host asynchronously on the same
       * stream as the SSYRK above to ensure it has finised updating the block
       * before it is copied */
      CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(B, ldb, 0, 0, A, lda, j, j, jb, jb, sizeof(float), stream0));
      /* Overlap the copy of the diagonal block with an out of place SGEMM (on a
       * different stream) which copies the column matrix below the diagonal
       * into a temporary matrix Dtemp */
      CU_ERROR_CHECK(cuSgemm2(handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, A + (j + jb) * sizeof(float), lda, A + j * sizeof(float), lda, one, A + (j * lda + j + jb) * sizeof(float), lda, Dtemp, dldd, stream1));
      CU_ERROR_CHECK(cuStreamSynchronize(stream0));     // Synchronise to ensure diagonal block is copied
      spotrf(CBlasLower, jb, B, ldb, info);             // before factorising on the host,
      if (*info != 0) {                                 // checking for positive definiteness
        *info += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }
      // Start copying the diagonal block back onto the device into the correct place
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0, jb, jb, sizeof(float), stream0));
      // and overlap that with an out-of-place inverse of the diagonal block via
      // the cholesky decomposition into Binv
      strtri2(CBlasLower, CBlasNonUnit, jb, B, ldb, Binv, ldb, info);
      // Copy the inverse into Binv on the device
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(dBinv, dldb, 0, 0, Binv, ldb, 0, 0, jb, jb, sizeof(float), stream0));
      CU_ERROR_CHECK(cuStreamSynchronize(stream1));     // Synchronise to ensure SGEMM is finished
      // Start the out-of-place triangular matrix multiply to copy Dtemp back
      // into A using the inverse of the diagonal block.  This is done on the
      // same stream as the copy to ensure it has completed first.
      CU_ERROR_CHECK(cuStrmm2(handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, dBinv, dldb, Dtemp, dldd, A + (j * lda + j + jb) * sizeof(float), lda, stream0));
    }
  }

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSpotrf(CUhandle * handles, int deviceCount, CBlasUplo uplo, size_t n, float * restrict A, size_t lda, long * restrict info) {
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
    spotrf(uplo, n, A, lda, info);
    return CUDA_SUCCESS;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUSsyrk(handles, deviceCount, CBlasUpper, CBlasTrans, jb, j, -one, &A[j * lda], lda, one, &A[j * lda + j], lda));
      spotf2(CBlasUpper, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handles, deviceCount, CBlasTrans, CBlasNoTrans, jb, n - j - jb, j, -one, &A[j * lda], lda, &A[(j + jb) * lda], lda, one, &A[(j + jb) * lda + j], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handles, deviceCount, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, jb, n - j - jb, one, &A[j * lda + j], lda, &A[(j + jb) * lda + j], lda));
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);

      CU_ERROR_CHECK(cuMultiGPUSsyrk(handles, deviceCount, CBlasLower, CBlasNoTrans, jb, j, -one, &A[j], lda, one, &A[j * lda + j], lda));
      spotf2(CBlasLower, jb, &A[j * lda + j], lda, info);
      if (*info != 0) {
        (*info) += (long)j;
        return CUDA_ERROR_INVALID_VALUE;
      }

      if (j + jb < n) {
        CU_ERROR_CHECK(cuMultiGPUSgemm(handles, deviceCount, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j, -one, &A[j + jb], lda, &A[j], lda, one, &A[j * lda + j + jb], lda));
        CU_ERROR_CHECK(cuMultiGPUStrsm(handles, deviceCount, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, n - j - jb, jb, one, &A[j * lda + j], lda, &A[j * lda + j + jb], lda));
      }
    }
  }

  return CUDA_SUCCESS;
}
