#include "lapack.h"
#include "error.h"

void cpotri(CBlasUplo uplo,
            size_t n,
            float complex * restrict A, size_t lda,
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

  ctrtri(uplo, CBlasNonUnit, n, A, lda, info);
  if (*info != 0)
    return;
  clauum(uplo, n, A, lda, info);
}

CUresult cuCpotri(CULAPACKhandle handle, CBlasUplo uplo,
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

  CU_ERROR_CHECK(cuCtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuClauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUCpotri(CUmultiGPULAPACKhandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          float complex * restrict A, size_t lda,
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

  CU_ERROR_CHECK(cuMultiGPUCtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuMultiGPUClauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}
