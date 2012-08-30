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

static inline CUresult cuMemcpyDtoD2DAsync(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                                          CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                                          size_t m, size_t n, size_t elemSize, CUstream stream) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2DAsync(&copy, stream);
}

static const float complex zero = 0.0f + 0.0f * I;
static const float complex one = 1.0f + 0.0f * I;

void ctrmm(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float complex alpha, const float complex * restrict A, size_t lda, float complex * restrict B, size_t ldb) {
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

  if (m == 0 || n == 0) return;

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
          for (size_t k = 0; k < m; k++) {
            if (B[j * ldb + k] != zero) {
              register float complex temp = alpha * B[j * ldb + k];
              for (size_t i = 0; i < k; i++)
                B[j * ldb + i] += temp * A[k * lda + i];
              if (diag == CBlasNonUnit) temp *= A[k * lda + k];
              B[j * ldb + k] = temp;
            }
          }
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          size_t k = m - 1;
          do {
            if (B[j * ldb + k] != zero) {
              register float complex temp = alpha * B[j * ldb + k];
              B[j * ldb + k] = temp;
              if (diag == CBlasNonUnit) B[j * ldb + k] *= A[k * lda + k];
              for (size_t i = k + 1; i < m; i++)
                B[j * ldb + i] -= temp * A[k * lda + i];
            }
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
            register float complex temp = B[j * ldb + i];
            if (transA == CBlasTrans) {
              if (diag == CBlasNonUnit) temp *= A[i * lda + i];
              for (size_t k = 0; k < i; k++)
                temp += A[i * lda + k] * B[j * ldb + k];
            }
            else {
              if (diag == CBlasNonUnit) temp *= conjf(A[i * lda + i]);
              for (size_t k = 0; k < i; k++)
                temp += conjf(A[i * lda + k]) * B[j * ldb + k];
            }
            B[j * ldb + i] = alpha * temp;
          } while (i-- > 0);
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            register float complex temp = B[j * ldb + i];
            if (transA == CBlasTrans) {
              if (diag == CBlasNonUnit) temp *= A[i * lda + i];
              for (size_t k = i + 1; k < m; k++)
                temp += A[i * lda + k] * B[j * ldb + k];
            }
            else {
              if (diag == CBlasNonUnit) temp *= conjf(A[i * lda + i]);
              for (size_t k = i + 1; k < m; k++)
                temp += conjf(A[i * lda + k]) * B[j * ldb + k];
            }
            B[j * ldb + i] = alpha * temp;
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
          register float complex temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            B[j * ldb + i] *= temp;
          for (size_t k = 0; k < j; k++) {
            if (A[j * lda + k] != zero) {
              register float complex temp = alpha * A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] += temp * B[k * ldb + i];
            }
          }
        } while (j-- > n);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          register float complex temp = alpha;
          if (diag == CBlasNonUnit) temp *= A[j * lda + j];
          for (size_t i = 0; i < m; i++)
            B[j * ldb + i] *= temp;
          for (size_t k = j + 1; k < n; k++) {
            if (A[j * lda + k] != zero) {
              register float complex temp = alpha * A[j * lda + k];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] += temp * B[k * ldb + i];
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
              register float temp = alpha * ((transA == CBlasTrans) ? A[k * lda + j] : conjf(A[k * lda + j]));
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] += temp * B[k * ldb + i];
            }
          }
          register float temp = alpha;
          if (diag == CBlasNonUnit) temp *= ((transA == CBlasTrans) ? A[k * lda + k] : conjf(A[k * lda + k]));
          if (temp != one) {
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= alpha;
          }
        }
      }
      else {
        size_t k = n - 1;
        do {
          for (size_t j = k + 1; j < n; j++) {
            if (A[k * lda + j] != zero) {
              register float temp = alpha * ((transA == CBlasTrans) ? A[k * lda + j] : conjf(A[k * lda + j]));
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] += temp * B[k * ldb + i];
            }
          }
          register float temp = alpha;
          if (diag == CBlasNonUnit) temp *= ((transA == CBlasTrans) ? A[k * lda + k] : conjf(A[k * lda + k]));
          if (temp != one) {
            for (size_t i = 0; i < m; i++)
              B[k * ldb + i] *= alpha;
          }
        } while (k-- > 0);
      }
    }
  }
}

CUresult cuCtrmm(CUmodule module, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float complex alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb, CUstream stream) {
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

  if (m == 0 || n == 0) return CUDA_SUCCESS;

  const unsigned int bx =  8;
  const unsigned int by =  8;
  const unsigned int mb = (side == CBlasLeft) ?  8 : 64;
  const unsigned int nb = (side == CBlasLeft) ? 64 :  8;

  char name[102];
  snprintf(name, 102, "_Z5ctrmmIL9CBlasSide%dEL9CBlasUplo%dEL14CBlasTranspose%dEL9CBlasDiag%dELj%uELj%uELj%uELj%uEEviifPKfiPfi", side, uplo, transA, diag, mb, nb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &m, &n, &alpha, &A, &lda, &B, &ldb };

  const unsigned int gx = (side == CBlasLeft) ? 1 : (unsigned int)max(1, (m + mb - 1) / mb);
  const unsigned int gy = (side == CBlasLeft) ? (unsigned int)max(1, (n + nb - 1) / nb) : 1;
  CU_ERROR_CHECK(cuLaunchKernel(function, gx, gy, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUCtrmm(CUcontext * contexts, int deviceCount, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float complex alpha, const float complex * restrict A, size_t lda, float complex * restrict B, size_t ldb) {
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

  if (m == 0 || n == 0) return CUDA_SUCCESS;

  if (alpha == zero) {
    cgemm(CBlasNoTrans, CBlasNoTrans, m, n, 0, zero, A, lda, B, ldb, zero, B, ldb);
    return CUDA_SUCCESS;
  }

  const size_t mb = (side == CBlasLeft) ?  8 : 16;
  const size_t nb = (side == CBlasLeft) ? 16 :  8;

  if (m <= mb || n <= nb) {
    ctrmm(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
    return CUDA_SUCCESS;
  }

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[(i + ib) * lda + i], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, i, -one, &A[i], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, transA, CBlasNoTrans, ib, jb, i, -one, &A[i * lda], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasLeft, CBlasUpper, transA, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, transA, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[i * lda + i + ib], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasLeft, CBlasLower, transA, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        } while (i > 0);
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, j, -one, &B[i], ldb, &A[j * lda], lda, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasRight, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[j * lda + j + jb], lda, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasRight, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, transA, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[(j + jb) * lda + j], lda, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasRight, CBlasUpper, transA, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUCgemm(contexts, deviceCount, CBlasNoTrans, transA, ib, jb, j, -one, &B[i], ldb, &A[j], lda, alpha, &B[j * ldb + i], ldb));

            ctrmm(CBlasRight, CBlasLower, transA, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        }
      }
    }
  }

  return CUDA_SUCCESS;
}
