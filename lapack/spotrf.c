#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "handle.h"
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
  if (n == 0)
    return CUDA_SUCCESS;

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
  if (uplo == CBlasUpper) {
    const unsigned int mb = 32;
    const unsigned int nb = 32;
    const unsigned int kb =  8;
    const unsigned int bx =  8;
    const unsigned int by =  8;

    if (jb > mb)
      return CUDA_ERROR_INVALID_VALUE;

    if (handle->spotrf == NULL)
      CU_ERROR_CHECK(cuModuleLoadData(&handle->spotrf, imageBytes));

    char name[67];
    snprintf(name, 67, "_Z9spotfimm2IL9CBlasUplo85ELj%uELj%uELj%uELj%uELj%uEEvPfS1_Piiiiii", mb, nb, kb, bx, by);

    CUfunction function;
    CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->spotrf, name));

    void * params[] = { &A, &B, &info, &lda, &ldb, &j, &jb, &n };

    const unsigned int gx = ((unsigned int)jb + nb - 1) / nb;
    const unsigned int gy = (unsigned int)((n - j - jb + nb - 1) / nb);

    CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)max(gx, 1), gy + 1, 1, bx, by, 1, 0, stream, params, NULL));
  }
  else {
    const unsigned int mb = 64;
    const unsigned int nb = 16;
    const unsigned int kb = 16;
    const unsigned int bx = 16;
    const unsigned int by =  4;

    if (jb > nb)
      return CUDA_ERROR_INVALID_VALUE;

    if (handle->spotrf == NULL)
      CU_ERROR_CHECK(cuModuleLoadData(&handle->spotrf, imageBytes));

    char name[67];
    snprintf(name, 67, "_Z9spotfimm2IL9CBlasUplo76ELj%uELj%uELj%uELj%uELj%uEEvPfS1_Piiiiii", mb, nb, kb, bx, by);

    CUfunction function;
    CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->spotrf, name));

    void * params[] = { &A, &B, &info, &lda, &ldb, &j, &jb, &n };

    const unsigned int gx = (unsigned int)((n - j - jb + mb - 1) / mb);
    const unsigned int gy = ((unsigned int)jb + nb - 1) / nb;

    CU_ERROR_CHECK(cuLaunchKernel(function, gx + 1, (unsigned int)max(gy, 1), 1, bx, by, 1, 0, stream, params, NULL));
  }
//   if ((uplo == CBlasUpper && jb > 32) || (uplo == CBlasLower && jb > 16))
//     return CUDA_ERROR_INVALID_VALUE;
//
//   if (handle->spotrf == NULL)
//     CU_ERROR_CHECK(cuModuleLoadData(&handle->spotrf, imageBytes));
//
//   /* For uplo == CBlasUpper m = jb, n = n - j - jb, k = j and an extra column
//    * of blocks is needed to compute the cholesky
//    * For uplo == CBlasLower m = n - j - jb, n = jb, k = j and an extra row of
//    * blocks is needed to compute the cholesky */
//   const unsigned int mb = (uplo == CBlasUpper) ? 32 : 64;
//   const unsigned int nb = (uplo == CBlasUpper) ? 32 : 16;
//   const unsigned int kb = (uplo == CBlasUpper) ?  8 : 16;
//   const unsigned int bx = (uplo == CBlasUpper) ?  8 : 16;
//   const unsigned int by = (uplo == CBlasUpper) ?  8 :  4;
//
//   char name[67];
//   snprintf(name, 67, "_Z9spotfimm2IL9CBlasUplo%dELj%uELj%uELj%uELj%uELj%uEEvPfS1_Piiiiii", uplo, mb, nb, kb, bx, by);
//
//   CUfunction function;
//   CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->spotrf, name));
//
//   void * params[] = { &A, &B, &info, &lda, &ldb, &j, &jb, &n };
//
//   const unsigned int gx = (uplo == CBlasUpper) ? ((unsigned int)jb + mb - 1) / mb + 1 : (unsigned int)max((n - j - jb + mb - 1) / mb, 1);
//   const unsigned int gy = (uplo == CBlasUpper) ? (unsigned int)max((n - j - jb + nb - 1) / nb, 1) : ((unsigned int)jb + nb - 1) / nb + 1;
//
//   CU_ERROR_CHECK(cuLaunchKernel(function, gx, gy, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuSpotrf(CULAPACKhandle handle,
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

  // Allocate the info parameter on the device
  CUdeviceptr dinfo;
  CU_ERROR_CHECK(cuMemAlloc(&dinfo, sizeof(int)));

  // If n <= 64 call the unblocked algorithm
  if (n <= 64) {
    CU_ERROR_CHECK(cuSpotf2(handle, uplo, n, A, lda, dinfo, NULL));
    int hinfo;
    CU_ERROR_CHECK(cuMemcpyDtoH(&hinfo, dinfo, sizeof(int)));
    CU_ERROR_CHECK(cuMemFree(dinfo));
    *info = hinfo;
    return CUDA_SUCCESS;
  }

  // (Maximum) dynamic block size
  size_t nb = n / 4;

  int hinfo = 0;
  float * D, * C;
  CUdeviceptr X;
  size_t ldb, ldc, ldx;
  CUstream stream0, stream1;

  // Allocate memory for diagonal blocks on host
  CU_ERROR_CHECK(cuMemAllocHost((void **)&D, (ldb = (n + 3u) & ~3u) * nb * sizeof(float)));
  CU_ERROR_CHECK(cuMemAllocHost((void **)&C, (ldc = (nb + 3u) & ~3u) * nb * sizeof(float)));

  // Create two streams for asynchronous copy and compute
  CU_ERROR_CHECK(cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING));

  if (uplo == CBlasUpper) {
    // Allocate temporary block row for out of place triangular multiply
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, nb * sizeof(float), n, sizeof(float)));
    ldx /= sizeof(float);

    // Decrease block size towards centre then increase
    for (size_t j = 0; j < n; j += nb, nb = (j < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using column matrix above */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasUpper, CBlasTrans, jb, j,
                             -one, A + j * lda * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));

      // If the block size is small enough
      if (jb <= 32) {
        // Call the combined SPOTF2/SGEMM kernel on the GPU to avoid host->device transfer
        CU_ERROR_CHECK(cuSpotfimm2(handle, uplo, j, jb, n, A + j * lda * sizeof(float), lda, X, ldx, dinfo, stream0));
        CU_ERROR_CHECK(cuMemcpyDtoH(&hinfo, dinfo, sizeof(int)));
        *info = hinfo;

        /* Check for positive definite matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }
      }
      else {

        // Work out whether it is worthwhile to do block column copy for the block size
        const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
        const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
        const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

        const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
        const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
        // Can only copy column back if the column was copied in the first place
        const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

        /* Out of place matrix multiply using column above and matrix to the right
        * to copy the row to the right of the diagonal block into X(0, nb) */
        CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasTrans, CBlasNoTrans,
                                jb, n - j - jb, j,
                                -one, A + j * lda * sizeof(float), lda,
                                A + (j + jb) * lda * sizeof(float), lda,
                                one, A + ((j + jb) * lda + j) * sizeof(float), lda,
                                X + nb * ldx * sizeof(float), ldx, stream1));

        /* Start copying diagonal block onto host asynchronously */
        float * B;
        if (bcc_dtoh) {
          // Copy the entire column in one go
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + j * lda * sizeof(float), n * jb * sizeof(float), stream0));
          B = &D[j];    // The diagonal block is half-way down
        }
        else {
          // Copy each column of the diagonal block separately
          CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, j, j,
                                            jb, jb, sizeof(float), stream0));
          B = D;      // The diagonal block is at the top of the column
        }

        /* Wait until the diagonal block has been copied */
        CU_ERROR_CHECK(cuStreamSynchronize(stream0));

        /* Perform the diagonal block decomposition using the CPU */
        spotrf(uplo, jb, B, ldb, info);

        /* Copy the diagonal block back onto the device */
        if (bcc_htod)
          CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), D, n * jb * sizeof(float), stream0));
        else
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, D, ldb, 0, 0,
                                            jb, jb, sizeof(float), stream0));

        /* Check for positive definite matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }

        /* While the diagonal block is copying compute the inverse out-of-place
        * (no need to check info for singularity as positive definite matrices
        * are always non-singular) */
        strtri2(uplo, CBlasNonUnit, jb, B, ldb, C, ldc, info);

        /* Copy the inverse onto the device in X */
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(X, ldx, 0, 0, C, ldc, 0, 0,
                                          jb, jb, sizeof(float), stream0));

      }

      /* Out of place triangular matrix multiply using the inverse of the
       * diagonal block and copying the temporary block row X back into the
       * correct place in A */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit,
                              jb, n - j - jb, one, X, ldx, X + nb * ldx * sizeof(float), ldx,
                              A + ((j + jb) * lda + j) * sizeof(float), lda, stream0));
    }
  }
  else {
    // Allocate temporary block column for out of place triangular multiply
    CU_ERROR_CHECK(cuMemAllocPitch(&X, &ldx, n * sizeof(float), nb, sizeof(float)));
    ldx /= sizeof(float);

    // Decrease block size towards centre then increase
    for (size_t j = 0; j < n; j += nb, nb = (j < n / 2) ? max(1, nb / 2) : min(n / 2, nb * 2)) {
      const size_t jb = min(nb, n - j);

      /* Rank-K update of diagonal block using row matrix to the left */
      CU_ERROR_CHECK(cuSsyrk(handle->blas_handle, CBlasLower, CBlasNoTrans, jb, j,
                             -one, A + j * sizeof(float), lda,
                             one, A + (j * lda + j) * sizeof(float), lda, stream0));

      // If the block size is small enough
      if (jb <= 16) {
        // Call the combined SPOTF2/SGEMM kernel on the GPU to avoid host->device transfer
        CU_ERROR_CHECK(cuSpotfimm2(handle, uplo, j, jb, n, A + j * sizeof(float), lda, X, ldx, dinfo, stream0));
        CU_ERROR_CHECK(cuMemcpyDtoH(&hinfo, dinfo, sizeof(int)));
        *info = hinfo;

        /* Check for positive definite matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }
      }
      else {

        // Work out whether it is worthwhile to do block column copy for the block size
        const double column_dtoh = (double)(n * jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH;
        const double block_dtoh = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_DTOH + OVERHEAD_DTOH);
        const bool bcc_dtoh = (lda == n && column_dtoh < block_dtoh);

        const double column_htod = (double)(n * jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD;
        const double block_htod = (double)jb * ((double)(jb * sizeof(float)) * BANDWIDTH_HTOD + OVERHEAD_HTOD);
        // Can only copy column back if the column was copied in the first place
        const bool bcc_htod = bcc_dtoh && (column_htod < block_htod);

        /* Out of place matrix multiply using row to the left and matrix below
        * to copy the column below of the diagonal block into X(nb, 0) */
        CU_ERROR_CHECK(cuSgemm2(handle->blas_handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, j,
                                -one, A + (j + jb) * sizeof(float), lda,
                                A + j * sizeof(float), lda,
                                one, A + (j * lda + j + jb) * sizeof(float), lda,
                                X + nb * sizeof(float), ldx, stream1));

        /* Start copying diagonal block onto host asynchronously */
        float * B;
        if (bcc_dtoh) {
          // Copy the entire column in one go
          CU_ERROR_CHECK(cuMemcpyDtoHAsync(D, A + j * lda * sizeof(float), n * jb * sizeof(float), stream0));
          B = &D[j];    // The diagonal block is half-way down
        }
        else {
          // Copy each column of the diagonal block separately
          CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(D, ldb, 0, 0, A, lda, j, j,
                                            jb, jb, sizeof(float), stream0));
          B = D;      // The diagonal block is at the top of the column
        }

        /* Wait until the diagonal block has been copied */
        CU_ERROR_CHECK(cuStreamSynchronize(stream0));

        /* Perform the diagonal block decomposition using the CPU */
        spotrf(uplo, jb, B, ldb, info);

        /* Copy the diagonal block back onto the device */
        if (bcc_htod)
          CU_ERROR_CHECK(cuMemcpyHtoDAsync(A + j * lda * sizeof(float), D, n * jb * sizeof(float), stream0));
        else
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A, lda, j, j, D, ldb, 0, 0,
                                            jb, jb, sizeof(float), stream0));

        /* Check for positive definite matrix */
        if (*info != 0) {
          *info += (long)j;
          break;
        }

        /* While the diagonal block is copying compute the inverse out-of-place
        * (no need to check info for singularity as positive definite matrices
        * are always non-singular) */
        strtri2(uplo, CBlasNonUnit, jb, B, ldb, C, ldc, info);

        /* Copy the inverse onto the device in X */
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(X, ldx, 0, 0, C, ldc, 0, 0,
                                          jb, jb, sizeof(float), stream0));

      }

      /* Out of place triangular matrix multiply using the inverse of the
       * diagonal block and copying the temporary block column X back into the
       * correct place in A */
      CU_ERROR_CHECK(cuStrmm2(handle->blas_handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit,
                              n - j - jb, jb, one, X, ldx, X + nb * sizeof(float), ldx,
                              A + (j * lda + j + jb) * sizeof(float), lda, stream0));
    }
  }

  // Clean up resources
  CU_ERROR_CHECK(cuMemFreeHost(D));
  CU_ERROR_CHECK(cuMemFreeHost(C));
  CU_ERROR_CHECK(cuMemFree(X));
  CU_ERROR_CHECK(cuMemFree(dinfo));

  CU_ERROR_CHECK(cuStreamDestroy(stream0));
  CU_ERROR_CHECK(cuStreamDestroy(stream1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSpotrf(CUmultiGPULAPACKhandle handle,
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
