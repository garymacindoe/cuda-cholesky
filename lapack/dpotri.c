#include "lapack.h"
#include "error.h"
#include "handle.h"

void dpotri(CBlasUplo uplo,
            size_t n,
            double * restrict A, size_t lda,
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

  dtrtri(uplo, CBlasNonUnit, n, A, lda, info);
  if (*info != 0)
    return;
  dlauum(uplo, n, A, lda, info);
}

CUresult cuDpotri(CULAPACKhandle handle,
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

  CU_ERROR_CHECK(cuDtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuDlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDpotri(CUmultiGPULAPACKhandle handle,
                          CBlasUplo uplo,
                          size_t n,
                          double * restrict A, size_t lda,
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

  CU_ERROR_CHECK(cuMultiGPUDtrtri(handle, uplo, CBlasNonUnit, n, A, lda, info));
  if (*info != 0)
    return CUDA_SUCCESS;
  CU_ERROR_CHECK(cuMultiGPUDlauum(handle, uplo, n, A, lda, info));
  return CUDA_SUCCESS;
}
