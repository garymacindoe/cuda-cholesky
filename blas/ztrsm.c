#include "blas.h"
#include "error.h"
#include <stdio.h>
#include "handle.h"
#include "config.h"
#include "ztrsm.fatbin.c"

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

static const double complex zero = 0.0 + 0.0 * I;
static const double complex one = 1.0 + 0.0 * I;

void ztrsm(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
           size_t m, size_t n,
           double complex alpha, const double complex * restrict A, size_t lda,
           double complex * restrict B, size_t ldb) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < nRowA)
    info = 9;
  else if (ldb < m)
    info = 11;
  if (info != 0) {
    XERBLA(info);
    return;
  }

  if (m == 0 || n == 0)
    return;

  if (alpha == zero) {
#pragma omp parallel for
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = zero;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= alpha;
          }
          size_t k = m - 1;
          do {
            if (B[j * ldb + k] != zero) {
              if (diag == CBlasNonUnit) B[j * ldb + k] /= A[k * lda + k];
              register double complex temp = B[j * ldb + k];
              for (size_t i = 0; i < k; i++)
                B[j * ldb + i] -= temp * A[k * lda + i];
            }
          } while (k-- > 0);
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= alpha;
          }
          for (size_t k = 0; k < m; k++) {
            if (B[j * ldb + k] != zero) {
              if (diag == CBlasNonUnit) B[j * ldb + k] /= A[k * lda + k];
              register double complex temp = B[j * ldb + k];
              for (size_t i = k + 1; i < m; i++)
                B[j * ldb + i] -= temp * A[k * lda + i];
            }
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            register double complex temp = alpha * B[j * ldb + i];
            if (transA == CBlasTrans) {
              for (size_t k = 0; k < i; k++)
                temp -= A[i * lda + k] * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            }
            else {
              for (size_t k = 0; k < i; k++)
                temp -= conj(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conj(A[i * lda + i]);
            }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            register double complex temp = alpha * B[j * ldb + i];
            if (transA == CBlasTrans) {
              for (size_t k = i + 1; k < m; k++)
                temp -= A[i * lda + k] * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            }
            else {
              for (size_t k = i + 1; k < m; k++)
                temp -= conj(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conj(A[i * lda + i]);
            }
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= alpha;
          }
          for (size_t k = 0; k < j; k++) {
            if (A[j * lda + k] != zero) {
              register double complex temp = A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] -= temp * B[k * ldb + i];
            }
          }
          if (diag == CBlasNonUnit) {
            register double complex temp = one / A[j * lda + j];
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= temp;
          }
        }
      }
      else {
        size_t j = n - 1;
        do {
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= alpha;
          }
          for (size_t k = j + 1; k < n; k++) {
            if (A[j * lda + k] != zero) {
              register double complex temp = A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] -= temp * B[k * ldb + i];
            }
          }
          if (diag == CBlasNonUnit) {
            register double complex temp = one / A[j * lda + j];
            for (size_t i = 0; i < m; i++)
              B[j * ldb + i] *= temp;
          }
        } while (j-- > 0);
      }
    }
    else {
      if (uplo == CBlasUpper) {
        size_t k = n - 1;
        do {
          if (diag == CBlasNonUnit) {
            register double complex temp;
            if (transA == CBlasTrans)
              temp = one / A[k * lda + k];
            else
              temp = one / conj(A[k * lda + k]);
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= temp;
          }
          for (size_t j = 0; j < k; j++) {
            if (A[k * lda + j] != zero) {
              register double complex temp;
              if (transA == CBlasTrans)
                temp = A[k * lda + j];
              else
                temp = conj(A[k * lda + j]);
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] -= temp * B[k * ldb + i];
            }
          }
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= alpha;
          }
        } while (k-- > 0);
      }
      else {
        for (size_t k = 0; k < n; k++) {
          if (diag == CBlasNonUnit) {
            register double complex temp;
            if (transA == CBlasTrans)
              temp = one / A[k * lda + k];
            else
              temp = one / conj(A[k * lda + k]);
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= temp;
          }
          for (size_t j = k + 1; j < n; j++) {
            if (A[k * lda + j] != zero) {
              register double complex temp;
              if (transA == CBlasTrans)
                temp = A[k * lda + j];
              else
                temp = conj(A[k * lda + j]);
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] -= temp * B[k * ldb + i];
            }
          }
          if (alpha != one) {
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= alpha;
          }
        }
      }
    }
  }
}

CUresult cuZtrsm(CUBLAShandle handle,
                 CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
                 size_t m, size_t n,
                 double complex alpha, CUdeviceptr A, size_t lda,
                 CUdeviceptr B, size_t ldb, CUstream stream) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < nRowA)
    info = 9;
  else if (ldb < m)
    info = 11;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0)
    return CUDA_SUCCESS;

  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->ztrsm == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->ztrsm, imageBytes));

  const unsigned int bx =  2;
  const unsigned int by =  2;
  const unsigned int mb = (side == CBlasLeft) ?  2 :  8;
  const unsigned int nb = (side == CBlasLeft) ?  8 :  2;

  char name[112];
  snprintf(name, 112,
           "_Z5ztrsmIL9CBlasSide%dEL9CBlasUplo%dEL14CBlasTranspose%dEL9CBlasDiag%dELj%uELj%uELj%uELj%uEEv7double2PKS4_PS4_iiii",
           side, uplo, transA, diag, mb, nb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->ztrsm, name));

  void * params[] = { &alpha, &A, &B, &lda, &ldb, &m, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  CU_ERROR_CHECK(cuCtxPopCurrent(&handle->context));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZtrsm(CUmultiGPUBLAShandle handle,
                         CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
                         size_t m, size_t n,
                         double complex alpha, const double complex * restrict A, size_t lda,
                         double complex * restrict B, size_t ldb) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < nRowA)
    info = 9;
  else if (ldb < m)
    info = 11;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0)
    return CUDA_SUCCESS;

  if (alpha == zero) {
    zgemm(CBlasNoTrans, CBlasNoTrans, m, n, 0, zero, A, lda, B, ldb, zero, B, ldb);
    return CUDA_SUCCESS;
  }

  const size_t mb = (transA == CBlasNoTrans) ? ZGEMM_N_MB : ZGEMM_CN_MB;
  const size_t nb = ZGEMM_N_NB;

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t r = m % mb;
        size_t i = (r == 0) ? m : m + mb - r;
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, ib, n, m - i - ib, -one, &A[(i + ib) * lda + i], lda, &B[i + ib], ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, ib, n, i, -one, &A[i], lda, B, ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        }
      }

    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, transA, CBlasNoTrans, ib, n, i, -one, &A[i * lda], lda, B, ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasLeft, CBlasUpper, transA, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        }
      }
      else {
        size_t r = m % mb;
        size_t i = (r == 0) ? m : m + mb - r;
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, transA, CBlasNoTrans, ib, n, m - i - ib, -one, &A[i * lda + i + ib], lda, &B[i + ib], ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasLeft, CBlasLower, transA, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        } while (i > 0);
      }

    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, m, jb, j, -one, B, ldb, &A[j * lda], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasRight, CBlasUpper, CBlasNoTrans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        }
      }
      else {
        size_t r = n % nb;
        size_t j = (r == 0) ? n : n + nb - r;
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, m, jb, n - j - jb, -one, &B[(j + jb) * ldb], ldb, &A[j * lda + j + jb], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasRight, CBlasLower, CBlasNoTrans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        } while (j > 0);
      }

    }
    else {
      if (uplo == CBlasUpper) {
        size_t r = n % nb;
        size_t j = (r == 0) ? n : n + nb - r;
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, transA, m, jb, n - j - jb, -one, &B[(j + jb) * ldb], ldb, &A[(j + jb) * lda + j], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasRight, CBlasUpper, transA, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, transA, m, jb, j, -one, B, ldb, &A[j], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle));
          ztrsm(CBlasRight, CBlasLower, transA, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        }
      }

    }
  }

  return CUDA_SUCCESS;
}
