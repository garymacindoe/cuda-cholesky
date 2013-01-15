#include "blas.h"
#include "error.h"
#include <stdio.h>
#include "handle.h"

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

static inline CUresult cuMemcpyDtoD2DAsync(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                                          CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static const double complex zero = 0.0 + 0.0 * I;
static const double complex one = 1.0 + 0.0 * I;

void ztrmm2(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag,
           size_t m, size_t n,
           double complex alpha, const double complex * restrict A, size_t lda,
           const double complex * restrict B, size_t ldb,
           double complex * restrict X, size_t ldx) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < nRowA)
    info = 9;
  else if (ldb < m)
    info = 11;
  else if (ldx < m)
    info = 13;
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
        X[j * ldx + i] = zero;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t k = 0; k < m; k++) {
            register double complex temp = B[j * ldb + k];
            if (B[j * ldb + k] != zero) {
              temp *= alpha;
              for (size_t i = 0; i < k; i++)
                X[j * ldx + i] += temp * A[k * lda + i];
              if (diag == CBlasNonUnit) temp *= A[k * lda + k];
            }
            X[j * ldx + k] = temp;
          }
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          size_t k = m - 1;
          do {
            if (B[j * ldb + k] != zero) {
              register double complex temp = alpha * B[j * ldb + k];
              X[j * ldx + k] = temp;
              if (diag == CBlasNonUnit) X[j * ldx + k] *= A[k * lda + k];
              for (size_t i = k + 1; i < m; i++)
                X[j * ldx + i] += temp * A[k * lda + i];
            }
            else
              X[j * ldx + k] = B[j * ldb + k];
          } while (k-- > 0);
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            register double complex temp = B[j * ldb + i];
            if (transA == CBlasTrans) {
              if (diag == CBlasNonUnit) temp *= A[i * lda + i];
              for (size_t k = 0; k < i; k++)
                temp += A[i * lda + k] * B[j * ldb + k];
            }
            else {
              if (diag == CBlasNonUnit) temp *= conj(A[i * lda + i]);
              for (size_t k = 0; k < i; k++)
                temp += conj(A[i * lda + k]) * B[j * ldb + k];
            }
            X[j * ldx + i] = alpha * temp;
          } while (i-- > 0);
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            register double complex temp = B[j * ldb + i];
            if (transA == CBlasTrans) {
              if (diag == CBlasNonUnit) temp *= A[i * lda + i];
              for (size_t k = i + 1; k < m; k++)
                temp += A[i * lda + k] * B[j * ldb + k];
            }
            else {
              if (diag == CBlasNonUnit) temp *= conj(A[i * lda + i]);
              for (size_t k = i + 1; k < m; k++)
                temp += conj(A[i * lda + k]) * B[j * ldb + k];
            }
            X[j * ldx + i] = alpha * temp;
          }
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t j = n - 1;
        do {
          register double complex temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            X[j * ldx + i] = temp * B[j * ldb + i];
          for (size_t k = 0; k < j; k++) {
            if (A[j * lda + k] != zero) {
              register double complex temp = alpha * A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          register double complex temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            X[j * ldx + i] = temp * B[j * ldb + i];
          for (size_t k = j + 1; k < n; k++) {
            if (A[j * lda + k] != zero) {
              register double complex temp = alpha * A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t k = 0; k < n; k++) {
          for (size_t j = 0; j < k; j++) {
            if (A[k * lda + j] != zero) {
              register double complex temp;
              if (transA == CBlasTrans)
                temp = alpha * A[k * lda + j];
              else
                temp = alpha * conj(A[k * lda + j]);
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
          register double complex temp = alpha;
          if (diag == CBlasNonUnit)
            temp *= ((transA == CBlasTrans) ? A[k * lda + k] : conj(A[k * lda + k]));
          if (temp != one) {
            for (size_t i = 0; i < m; i++)
              X[k * ldx + i] = temp * B[k * ldb + i];
          }
        }
      }
      else {
        size_t k = n - 1;
        do {
          for (size_t j = k + 1; j < n; j++) {
            if (A[k * lda + j] != zero) {
              register double complex temp;
              if (transA == CBlasTrans)
                temp = alpha * A[k * lda + j];
              else
                temp = alpha * conj(A[k * lda + j]);
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
          register double complex temp = alpha;
          if (diag == CBlasNonUnit)
            temp *= ((transA == CBlasTrans) ? A[k * lda + k] : conj(A[k * lda + k]));
          if (temp != one) {
            for (size_t i = 0; i < m; i++)
              X[k * ldx + i] = temp * B[k * ldb + i];
          }
        } while (k-- > 0);
      }
    }
  }
}

CUresult cuZtrmm2(CUmodule module,
                  CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
                  size_t m, size_t n,
                  double complex alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
                  CUdeviceptr X, size_t ldx, CUstream stream) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < nRowA)
    info = 9;
  else if (ldb < m)
    info = 11;
  else if (ldx < m)
    info = 13;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0)
    return CUDA_SUCCESS;

  const unsigned int mb = (side == CBlasRight) ? 64 :  8;
  const unsigned int nb = (side == CBlasRight) ?  4 :  8;
  const unsigned int kb = (side == CBlasRight) ? 16 :  4;
  const unsigned int bx = (side == CBlasRight) ? 16 :  4;
  const unsigned int by = (side == CBlasRight) ?  4 :  8;

  char name[95];
  if (trans == CBlasNoTrans)
    snprintf(name, 78,
             "_Z8ztrmm%c%c%cIL9CBlasDiag%dELj%uELj%uELj%uELj%uELj%uEEvPK7double2S3_PS1_S1_iiiii",
             side, uplo, trans, diag, mb, nb, kb, bx, by);
  else
    snprintf(name, 95,
             "_Z8ztrmm%c%cTIL14CBlasTranspose%dEL9CBlasDiag%dELj%uELj%uELj%uELj%uELj%uEEvPK7double2S4_PS2_S2_iiiii",
             side, uplo, trans, diag, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &alpha, &A, &B, &X, &lda, &ldb, &ldx, &m, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function,
                                (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1,
                                0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZtrmm(CUmultiGPUBlasHandle handle,
                         CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
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

  const size_t mb = (trans == CBlasNoTrans) ? ZGEMM_N_MB : ZGEMM_CN_MB;
  const size_t nb = ZGEMM_N_NB;

  if (m <= mb || n <= nb) {
    ztrmm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    return CUDA_SUCCESS;
  }

  if (side == CBlasLeft) {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, ib, n, m - i - ib, -one, &A[(i + ib) * lda + i], lda, &B[i + ib], ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, ib, n, i, -one, &A[i], lda, B, ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, trans, CBlasNoTrans, ib, n, i, -one, &A[i * lda], lda, B, ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasLeft, CBlasUpper, trans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        }
      }
      else {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, trans, CBlasNoTrans, ib, n, m - i - ib, -one, &A[i * lda + i + ib], lda, &B[i + ib], ldb, alpha, &B[i], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasLeft, CBlasLower, trans, diag, ib, n, one, &A[i * lda + i], lda, &B[i], ldb);
        } while (i > 0);
      }
    }
  }
  else {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, m, jb, j, -one, B, ldb, &A[j * lda], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasRight, CBlasUpper, CBlasNoTrans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        }
      }
      else {
        size_t j = (n + nb - 1) & ~(nb - 1);
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, CBlasNoTrans, m, jb, n - j - jb, -one, &B[(j + jb) * ldb], ldb, &A[j * lda + j + jb], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasRight, CBlasLower, CBlasNoTrans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        } while (j > 0);
      }
    }
    else {
      if (uplo == CBlasUpper) {
        size_t j = (n + nb - 1) & ~(nb - 1);
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, trans, m, jb, n - j - jb, -one, &B[(j + jb) * ldb], ldb, &A[(j + jb) * lda + j], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasRight, CBlasUpper, trans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);
          CU_ERROR_CHECK(cuMultiGPUZgemm(handle, CBlasNoTrans, trans, m, jb, j, -one, B, ldb, &A[j], lda, alpha, &B[j * ldb], ldb));
          CU_ERROR_CHECK(cuMultiGPUBlasSynchronize(handle));
          ztrmm(CBlasRight, CBlasLower, trans, diag, m, jb, one, &A[j * lda + j], lda, &B[j * ldb], ldb);
        }
      }
    }
  }

  return CUDA_SUCCESS;
}
