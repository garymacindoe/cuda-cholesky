#include "lapack.h"
#include "handle.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "config.h"
#include "spotrf.fatbin.c"

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

static inline CUresult cuSpotfimm2(CULAPACKhandle handle, CBlasUplo uplo,
                                   size_t j, size_t jb, size_t n,
                                   CUdeviceptr A, size_t lda,
                                   CUdeviceptr B, size_t ldb,
                                   CUdeviceptr info, CUstream stream) {
  if (handle->spotrf == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->spotrf, imageBytes));

  /* For uplo == CBlasUpper m = jb, n = n - j - jb, k = j and an extra column
   * of blocks is needed to compute the cholesky
   * For uplo == CBlasLower m = n - j - jb, n = jb, k = j and an extra row of
   * blocks is needed to compute the cholesky */
  const unsigned int mb = (uplo == CBlasUpper) ? 32 : 64;
  const unsigned int nb = (uplo == CBlasUpper) ? 32 : 16;
  const unsigned int kb = (uplo == CBlasUpper) ?  8 : 16;
  const unsigned int bx = (uplo == CBlasUpper) ?  8 : 16;
  const unsigned int by = (uplo == CBlasUpper) ?  8 :  4;

  char name[67];
  snprintf(name, 67, "_Z9spotfimm2IL9CBlasUplo%dELj%uELj%uELj%uELj%uELj%uEEvPfS1_Piiiiii", uplo, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->spotrf, name));

  void * params[] = { &A, &B, &info, &lda, &ldb, &j, &jb, &n };

  const unsigned int gx = (uplo == CBlasUpper) ? ((unsigned int)jb + mb - 1) / mb : (unsigned int)(n - j - jb + mb - 1) / mb + 1;
  const unsigned int gy = (uplo == CBlasUpper) ? (unsigned int)(n - j - jb + nb - 1) / nb + 1 : ((unsigned int)jb + nb - 1) / nb;

  CU_ERROR_CHECK(cuLaunchKernel(function, gx, gy, 1, bx, by, 1, 0, stream, params, NULL));

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

  // Allocate the info parameter on the device
  CUdeviceptr dinfo;
  CU_ERROR_CHECK(cuMemAlloc(&dinfo, sizeof(long)));

  // If n <= 64 call the unblocked algorithm
  if (n <= 64) {
    CU_ERROR_CHECK(cuSpotf2(handle, uplo, n, A, lda, dinfo, NULL));
    CU_ERROR_CHECK(cuMemcpyDtoH(info, dinfo, sizeof(long)));
    CU_ERROR_CHECK(cuMemFree(dinfo));
    return CUDA_SUCCESS;
  }

  // Find out what the current GPU can do
  CUdevice device;
  CU_ERROR_CHECK(cuCtxGetDevice(&device));

  int concurrentCopies, concurrentKernels;
  // Find out if the current GPU can overlap memory copies with kernel execution
  CU_ERROR_CHECK(cuDeviceGetAttribute(&concurrentCopies, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device));
  // Find out if the current GPU can execute multiple kernels simultaneously
  CU_ERROR_CHECK(cuDeviceGetAttribute(&concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device));

  // static block size
//   const size_t nb = (uplo == CBlasUpper) ? 256 : 128;
  // dynamic block sizing
  size_t nb = n / 4;

  float * B, * C, * X;
  size_t ldb, ldc;
  CUdeviceptr D;
  size_t ldd;
  CUstream stream0, stream1;

  // Allocate memory for the block column
  CU_ERROR_CHECK(cuMemAllocHost((void **)&X, (ldb = n) * nb * sizeof(float)));
  B = X;
  CU_ERROR_CHECK(cuMemAllocHost((void **)&C, (ldc = (nb + 3u) & ~3u) * nb * sizeof(float)));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, 0));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, 0));

  if (uplo == CBlasUpper) {
    // Allocate a temporary row matrix for out of place STRTRI, SGEMM and STRMM */
    CU_ERROR_CHECK(cuMemAllocPitch(&D, &ldd, nb * sizeof(float), n, sizeof(float)));
    ldd /= sizeof(float);

    // Static block sizing
//     for (size_t j = 0; j < n; j += nb) {
    // Dynamic block sizing - nb = nb/2 up to n / 2 then nb = n*2 up to n
    for (size_t j = 0; j < n; j += nb, nb = (j < n / 2) ? max(1, nb / 2) : min(n, nb * 2)) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                             -one, A + j * lda * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));
      /* If nb <= 32 then the entire iteration can be performed on the GPU */
      /* (the SGEMM kernel processes D in blocks of 32x32 for D = -CA^T + D) */
      if (nb <= 32) {
        if (concurrentKernels != 0) {
          /* If the GPU supports concurrent kernels execute the two kernels on
           * different streams and have the GPU execute them simultaneously */
          CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasTrans, CBlasNoTrans,
                                  jb, n - j - jb, j,
                                  -one, A + j * lda * sizeof(float), lda,
                                  A + (j + jb) * lda * sizeof(float), lda,
                                  one, A + ((j + jb) * lda + j) * sizeof(float), lda,
                                  D + nb * ldd * sizeof(float), ldd, stream1));
          /* Do the SPOTF2 and STRTI2 on the same stream as the SSYRK and STRMM */
          CU_ERROR_CHECK(cuSpotf2(handle, uplo, jb, A, lda, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for positive definite matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
          CU_ERROR_CHECK(cuStrti22(handle, uplo, CBlasNonUnit, jb, A, lda, D, ldd, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for non-singular matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
        }
        else {
          /* Call a combined kernel which overlaps the unblocked cholesky and
           * inverse with matrix multply on the GPU */
          CU_ERROR_CHECK(cuSpotfimm2(handle, uplo, j, jb, n, A + j * lda * sizeof(float), lda, D, ldd, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for positive definite/non-singular matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
        }
      }
      else {
      /* Overlap the SSYRK with an out of place SGEMM (on a different stream)
       * which copies the row to the right of the diagonal into D(0,nb) */
      CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasTrans, CBlasNoTrans,
                              jb, n - j - jb, j,
                              -one, A + j * lda * sizeof(float), lda,
                              A + (j + jb) * lda * sizeof(float), lda,
                              one, A + ((j + jb) * lda + j) * sizeof(float), lda,
                              D + nb * ldd * sizeof(float), ldd, stream1));

        // Work out whether it is worthwhile to do block column copy for the block size
        const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
        const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
        const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

        const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
        const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
        // Can only copy column back if the column was copied in the first place
        const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

        /* Start copying diagonal block onto host asynchronously on the same
        * stream as the SSYRK to ensure it has finised updating the block
        * before it is copied */
        if (bcc_dtoh) {
          // Copy the entire column
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(X, A + j * lda * sizeof(float), n * jb * sizeof(float), stream0));
          B = &X[j];    // The diagonal block is half-way down
        }
        else
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
        if (bcc_htod)
          CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), X, n * jb * sizeof(float), stream0));
        else
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                            jb, jb, sizeof(float), stream0));
        /* Calculate the inverse out of place on the CPU while the diagonal block
        * is being copied by the device */
        strtri2(CBlasUpper, CBlasNonUnit, jb, B, ldb, C, ldc, info);
        /* Check for singular matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }
        /* Copy the inverse back onto the device into the left block of D */
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(D, ldd, 0, 0, C, ldc, 0, 0,
                                          jb, jb, sizeof(float), stream0));
      }
      /* Wait until the SGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
      if (concurrentKernels != 0)
        CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Perform the out of place triangular matrix multiply to copy D back into
       * the correct place */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit,
                              jb, n - j - jb, one, D, ldd, D + nb * ldd * sizeof(float), ldd,
                              A + ((j + jb) * lda + j) * sizeof(float), lda, stream0));
    }
  }
  else {
    // Allocate a temporary column matrix for out of place STRTRI, SGEMM and STRMM */
    CU_ERROR_CHECK(cuMemAllocPitch(&D, &ldd, n * sizeof(float), nb, sizeof(float)));
    ldd /= sizeof(float);

    // Static block sizing
//     for (size_t j = 0; j < n; j += nb) {
    // Dynamic block sizing - nb = nb/2 up to n / 2 then nb = n*2 up to n
    for (size_t j = 0; j < n; j += nb, nb = (j < n / 2) ? max(1, nb / 2) : min(n, nb * 2)) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                             -one, A + j * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));

      /* If nb <= 16 then the entire iteration can be performed on the GPU */
      /* (the SGEMM kernel processes D in blocks of 64x16 for D = -AC^T + D) */
      if (nb <= 16) {
        if (concurrentKernels != 0) {
          /* If the GPU supports concurrent kernels execute the two kernels on
           * different streams and have the GPU execute them simultaneously */
          CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                                  -one, A + (j + jb) * sizeof(float), lda,
                                  A + j * sizeof(float), lda,
                                  one, A + (j * lda + j + jb) * sizeof(float), lda,
                                  D + nb * sizeof(float), ldd, stream1));
          /* Do the SPOTF2 and STRTI2 on the same stream as the SSYRK and STRMM */
          CU_ERROR_CHECK(cuSpotf2(handle, uplo, jb, A, lda, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for positive definite matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
          CU_ERROR_CHECK(cuStrti22(handle, uplo, CBlasNonUnit, jb, A, lda, D, ldd, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for non-singular matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
        }
        else {
          /* Call a combined kernel which overlaps the unblocked cholesky and
           * inverse with matrix multply on the GPU */
          CU_ERROR_CHECK(cuSpotfimm2(handle, uplo, j, jb, n, A + j * sizeof(float), lda, D, ldd, dinfo, stream0));
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(info, dinfo, sizeof(long), stream0));
          CU_ERROR_CHECK(cuStreamSynchronize(stream0));
          /* Check for positive definite/non-singular matrix */
          if (*info != 0) {
            *info += (long)j;
            break;
          }
        }
      }
      else {
        /* Overlap the SSYRK with an out of place SGEMM (on a different stream)
        * which copies the column under of the diagonal into D(nb,0) */
        CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                                -one, A + (j + jb) * sizeof(float), lda,
                                A + j * sizeof(float), lda,
                                one, A + (j * lda + j + jb) * sizeof(float), lda,
                                D + nb * sizeof(float), ldd, stream1));

        // Work out whether it is worthwhile to do block column copy for the block size
        const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
        const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
        const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

        const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
        const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
        // Can only copy column back if the column was copied in the first place
        const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

        /* Start copying diagonal block onto host asynchronously on the same
        * stream as the SSYRK to ensure it has finised updating the block
        * before it is copied */
        if (bcc_dtoh) {
          // Copy the entire column
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(X, A + j * lda * sizeof(float), n * jb * sizeof(float), stream0));
          B = &X[j];    // The diagonal block is half-way down
        }
        else
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
        if (bcc_htod)
          CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), X, n * jb * sizeof(float), stream0));
        else
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, B, ldb, 0, 0,
                                            jb, jb, sizeof(float), stream0));
        /* Calculate the inverse out of place on the CPU while the diagonal block
        * is being copied by the device */
        strtri2(CBlasLower, CBlasNonUnit, jb, B, ldb, C, ldc, info);
        /* Check for singular matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }
        /* Copy the inverse back onto the device into the left block of D */
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(D, ldd, 0, 0, C, ldc, 0, 0,
                                          jb, jb, sizeof(float), stream0));
      }
      /* Wait until the SGEMM has finished updating the row to the right (this
       * is unnecessary on devices that cannot execute multiple kernels
       * simultaneously */
      if (concurrentKernels != 0)
        CU_ERROR_CHECK(cuStreamSynchronize(stream1));
      /* Triangular solve of the diagonal block using the column matrix below
       * on the same stream as the copy to ensure it has completed first */
      /* Perform the out of place triangular matrix multiply to copy D back into
       * the correct place */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit,
                              n - j - jb, jb, one, D, ldd, D + nb * sizeof(float), ldd,
                              A + (j * lda + j + jb) * sizeof(float), lda, stream0));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(X));
  CU_ERROR_CHECK(cuMemFreeHost(C));
  CU_ERROR_CHECK(cuMemFree(D));
  CU_ERROR_CHECK(cuMemFree(dinfo));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
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
