#include "lapack.h"
#include "error.h"

void zpotri(CBlasUplo uplo,
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

  if (n == 0)
    return;

  ztrtri(uplo, CBlasNonUnit, n, A, lda, info);
  if (*info != 0)
    return;
  zlauum(uplo, n, A, lda, info);
}

CUresult cuZpotri(CUblashandle handle,
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

  CU_ERROR_CHECK(cuZtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuZlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZpotri(CUmultiGPUBlasHandle handle,
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

  CU_ERROR_CHECK(cuMultiGPUZtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuMultiGPUZlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}
