#include "blas.h"
#include "error.h"
#include <stdio.h>

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

static const double zero = 0.0;
static const double one = 1.0;

void dtrmm2(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
            size_t m, size_t n,
            double alpha, const double * restrict A, size_t lda,
            const double * restrict B, size_t ldb,
            double * restrict X, size_t ldx) {
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
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t k = 0; k < m; k++) {
            register double temp = B[j * ldb + k];
            if (temp != zero) {
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
              register double temp = alpha * B[j * ldb + k];
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
            register double temp = B[j * ldb + i];
            if (diag == CBlasNonUnit) temp *= A[i * lda + i];
            for (size_t k = 0; k < i; k++)
              temp += A[i * lda + k] * B[j * ldb + k];
            X[j * ldx + i] = alpha * temp;
          } while (i-- > 0);
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            register double temp = B[j * ldb + i];
            if (diag == CBlasNonUnit) temp *= A[i * lda + i];
            for (size_t k = i + 1; k < m; k++)
              temp += A[i * lda + k] * B[j * ldb + k];
            X[j * ldx + i] = alpha * temp;
          }
        }
      }
    }
  }
  else {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t j = n - 1;
        do {
          register double temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            X[j * ldx + i] = temp * B[j * ldb + i];
          for (size_t k = 0; k < j; k++) {
            if (A[j * lda + k] != zero) {
              register double temp = alpha * A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          register double temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            X[j * ldx + i] = temp * B[j * ldb + i];
          for (size_t k = j + 1; k < n; k++) {
            if (A[j * lda + k] != zero) {
              register double temp = alpha * A[j * lda + k];
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
              register double temp = alpha * A[k * lda + j];
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
          register double temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[k * lda + k];
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
              register double temp = alpha * A[k * lda + j];
              for (size_t i = 0; i < m; i++)
                X[j * ldx + i] += temp * B[k * ldb + i];
            }
          }
          register double temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[k * lda + k];
          if (temp != one) {
            for (size_t i = 0; i < m; i++)
              X[k * ldx + i] = temp * B[k * ldb + i];
          }
        } while (k-- > 0);
      }
    }
  }
}

CUresult cuDtrmm2(CUmodule module,
                  CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
                  size_t m, size_t n,
                  double alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
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

  const unsigned int mb = (side == CBlasLeft && trans != CBlasNoTrans) ? 32 : 64;
  const unsigned int nb = (side == CBlasLeft && trans != CBlasNoTrans) ? 16 :  8;
  const unsigned int kb = (side == CBlasLeft && trans != CBlasNoTrans) ?  8 : 16;
  const unsigned int bx = (trans == CBlasNoTrans) ? 16 :  8;
  const unsigned int by = (trans == CBlasNoTrans) ?  4 :  8;

  char name[100];
  snprintf(name, 100,
           "_Z7dtrmm2%cIL9CBlasUplo%dEL14CBlasTranspose%dEL9CBlasDiag%dELj%uELj%uELj%uELj%uELj%uEEviidPKdiS4_iPdi",
           side, uplo, trans, diag, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &m, &n, &alpha, &A, &lda, &B, &ldb, &X, &ldx };

  CU_ERROR_CHECK(cuLaunchKernel(function,
                                (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1,
                                0, stream, params, NULL));

  return CUDA_SUCCESS;
}
#if 0
CUresult cuMultiGPUDtrmm(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
                         size_t m, size_t n,
                         double alpha, const double * restrict A, size_t lda,
                         double * restrict B, size_t ldb) {
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
    dgemm(CBlasNoTrans, CBlasNoTrans, m, n, 0, zero, A, lda, B, ldb, zero, B, ldb);
    return CUDA_SUCCESS;
  }

  const size_t mb = (side == CBlasLeft) ?  8 : 16;
  const size_t nb = (side == CBlasLeft) ? 16 :  8;

  if (m <= mb || n <= nb) {
    dtrmm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    return CUDA_SUCCESS;
  }

  if (side == CBlasLeft) {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[(i + ib) * lda + i], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, i, -one, &A[i], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasTrans, CBlasNoTrans, ib, jb, i, -one, &A[i * lda], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasLeft, CBlasUpper, CBlasTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        }
      }
      else {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[i * lda + i + ib], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasLeft, CBlasLower, CBlasTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        } while (i > 0);
      }
    }
  }
  else {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, j, -one, &B[i], ldb, &A[j * lda], lda, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasRight, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        }
      }
      else {
        size_t j = (n + nb - 1) & ~(nb - 1);
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[j * lda + j + jb], lda, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasRight, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        } while (j > 0);
      }
    }
    else {
      if (uplo == CBlasUpper) {
        size_t j = (n + nb - 1) & ~(nb - 1);
        do {
          j -= nb;
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasTrans, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[(j + jb) * lda + j], lda, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasRight, CBlasUpper, CBlasTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUDgemm(handles, deviceCount, CBlasNoTrans, CBlasTrans, ib, jb, j, -one, &B[i], ldb, &A[j], lda, alpha, &B[j * ldb + i], ldb));

            dtrmm(CBlasRight, CBlasLower, CBlasTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        }
      }
    }
  }

  return CUDA_SUCCESS;
}
#endif
