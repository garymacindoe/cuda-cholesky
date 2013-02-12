#include "lapack.h"
#include "error.h"

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

CUresult cuSpotri(CUBLAShandle handle,
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

  CU_ERROR_CHECK(cuStrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuSlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSpotri(CUmultiGPUBLAShandle handle,
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

  CU_ERROR_CHECK(cuMultiGPUStrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuMultiGPUSlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}
