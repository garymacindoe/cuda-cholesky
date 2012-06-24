#include "blas.h"
#include "error.h"
#include <stdio.h>

static inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }
static inline size_t max(size_t a, size_t b) { return (a > b) ? a : b; }
static inline unsigned int maxj(unsigned int a, unsigned int b) { return (a > b) ? a : b; }

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

static const float zero = 0.0f;
static const float one = 1.0f;

// #ifdef MKL_ILP64
//   extern void strsm_(const char *, const char *, const char *, const char *, const long *, const long *, const float *, const float *, const long *, float *, const long *);
// #else
//   extern void strsm_(const char *, const char *, const char *, const char *, const int *, const int *, const float *, const float *, const int *, float *, const int *);
// #endif
void strsm(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float alpha, const float * A, size_t lda, float * B, size_t ldb) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
// #ifdef MKL_ILP64
//   strsm_((const char *)&side, (const char *)&uplo, (const char *)&transA, (const char *)&diag, (const long *)&m, (const long *)&n, &alpha, A, (const long *)&lda, B, (const long *)&ldb);
// #else
//   strsm_((const char *)&side, (const char *)&uplo, (const char *)&transA, (const char *)&diag, (const int *)&m, (const int *)&n, &alpha, A, (const int *)&lda, B, (const int *)&ldb);
// #endif
//   return;
  if (lda < max(1, nRowA))
    info = 9;
  else if (ldb < max(1, m))
    info = 11;
  if (info != 0)
    XERBLA(info);

  if (m == 0 || n == 0) return;

  if (alpha == zero) {
#pragma omp parallel for
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = zero;
    }
    return;
  }

  // SSE 4.1 has a dot product (x(0)*y(0) + x(1)*y(1) + x(2)*y(2) + x(3)*y(3))
  // instruction but not a dot product subtraction (x(0)*y(0) - x(1)*y(1) -
  // x(2)*y(2) - x(3)*y(3)) which is why some of the loops are written oddly
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
              register float temp = B[j * ldb + k];
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
              register float temp = B[j * ldb + k];
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
            register float temp = zero;
            for (size_t k = 0; k < i; k++)
              temp += A[i * lda + k] * B[j * ldb + k];
            temp = alpha * B[j * ldb + i] - temp;
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            register float temp = zero;
            for (size_t k = i + 1; k < m; k++)
              temp += A[i * lda + k] * B[j * ldb + k];
            temp = alpha * B[j * ldb + i] - temp;
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
#ifdef _OPENMP
#pragma omp parallel for
        for (size_t _i = 0; _i < m; _i += 8) {
          const size_t _m = min(8, m - _i);
          float * restrict _B = &B[_i];
#define B _B
#define m _m
#endif
          for (size_t j = 0; j < n; j++) {
            if (alpha != one) {
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] *= alpha;
            }
            for (size_t k = 0; k < j; k++) {
              if (A[j * lda + k] != zero) {
                register float temp = A[j * lda + k];
                for (size_t i = 0; i < m; i++)
                  B[j * ldb + i] -= temp * B[k * ldb + i];
              }
            }
            if (diag == CBlasNonUnit) {
              register float temp = one / A[j * lda + j];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] *= temp;
            }
          }
#ifdef _OPENMP
#undef B
#undef m
        }
#endif
      }
      else {
#ifdef _OPENMP
#pragma omp parallel for
        for (size_t _i = 0; _i < m; _i += 8) {
          const size_t _m = min(8, m - _i);
          float * restrict _B = &B[_i];
#define B _B
#define m _m
#endif
          size_t j = n - 1;
          do {
            if (alpha != one) {
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] *= alpha;
            }
            for (size_t k = j + 1; k < n; k++) {
              if (A[j * lda + k] != zero) {
                register float temp = A[j * lda + k];
                for (size_t i = 0; i < m; i++)
                  B[j * ldb + i] -= temp * B[k * ldb + i];
              }
            }
            if (diag == CBlasNonUnit) {
              register float temp = one / A[j * lda + j];
              for (size_t i = 0; i < m; i++)
                B[j * ldb + i] *= temp;
            }
          } while (j-- > 0);
#ifdef _OPENMP
#undef B
#undef m
        }
#endif
      }
    }
    else {
      if (uplo == CBlasUpper) {
#ifdef _OPENMP
#pragma omp parallel for
        for (size_t _i = 0; _i < m; _i += 8) {
          const size_t _m = min(8, m - _i);
          float * restrict _B = &B[_i];
#define B _B
#define m _m
#endif
          size_t k = n - 1;
          do {
            if (diag == CBlasNonUnit) {
              register float temp = one / A[k * lda + k];
              for (size_t i = 0; i < m; i++)
                B[k * ldb + i] *= temp;
            }
            for (size_t j = 0; j < k; j++) {
              if (A[k * lda + j] != zero) {
                register float temp = A[k * lda + j];
                for (size_t i = 0; i < m; i++)
                  B[j * ldb + i] -= temp * B[k * ldb + i];
              }
            }
            if (alpha != one) {
              for (size_t i = 0; i < m; i++)
                B[k * ldb + i] *= alpha;
            }
          } while (k-- > 0);
#ifdef _OPENMP
#undef B
#undef m
        }
#endif
      }
      else {
#ifdef _OPENMP
#pragma omp parallel for
        for (size_t _i = 0; _i < m; _i += 8) {
          const size_t _m = min(8, m - _i);
          float * restrict _B = &B[_i];
#define B _B
#define m _m
#endif
          for (size_t k = 0; k < n; k++) {
            if (diag == CBlasNonUnit) {
              register float temp = one / A[k * lda + k];
              for (size_t i = 0; i < m; i++)
                B[k * ldb + i] *= temp;
            }
            for (size_t j = k + 1; j < n; j++) {
              if (A[k * lda + j] != zero) {
                register float temp = A[k * lda + j];
                for (size_t i = 0; i < m; i++)
                  B[j * ldb + i] -= temp * B[k * ldb + i];
              }
            }
            if (alpha != one) {
              for (size_t i = 0; i < m; i++)
                B[k * ldb + i] *= alpha;
            }
          }
#ifdef _OPENMP
#undef B
#undef m
        }
#endif
      }
    }
  }
}

static inline CUresult cuStrsm2(CUmodule module, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb, CUstream stream) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 9;
  else if (ldb < max(1, m))
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
  snprintf(name, 102, "_Z5strsmIL9CBlasSide%dEL9CBlasUplo%dEL14CBlasTranspose%dEL9CBlasDiag%dELj%uELj%uELj%uELj%uEEviifPKfiPfi", side, uplo, transA, diag, mb, nb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &m, &n, &alpha, &A, &lda, &B, &ldb };

  const unsigned int gx = (side == CBlasLeft) ? 1 : maxj(1, (unsigned int)(m + mb - 1) / mb);
  const unsigned int gy = (side == CBlasLeft) ? maxj(1, (unsigned int)(n + nb - 1) / nb) : 1;
  CU_ERROR_CHECK(cuLaunchKernel(function, gx, gy, 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuStrsm(CUmodule module, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb, CUstream stream) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 9;
  else if (ldb < max(1, m))
    info = 11;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0) return CUDA_SUCCESS;

  CUmodule sgemm;

  if (alpha == zero) {
    CU_ERROR_CHECK(cuModuleLoad(&sgemm, "sgemm.cubin"));
    CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasNoTrans, m, n, 0, zero, A, lda, B, ldb, zero, B, ldb, stream));
    CU_ERROR_CHECK(cuModuleUnload(sgemm));
    return CUDA_SUCCESS;
  }

  const size_t mb = (side == CBlasLeft) ?  256 : 3840;
  const size_t nb = (side == CBlasLeft) ? 3840 :  256;

  if (m <= mb || n <= nb)
    return cuStrsm2(module, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, stream);

  CU_ERROR_CHECK(cuModuleLoad(&sgemm, "sgemm.cubin"));

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        size_t i = (m + mb - 1) & ~(mb - 1);
        do {
          i -= mb;
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, A + ((i + ib) * lda + i) * sizeof(float), lda, B + (j * ldb + i + ib) * sizeof(float), ldb, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, A + (i * lda + i) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
          }
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasNoTrans, ib, jb, i, -one, A + i * sizeof(float), lda, B + j * ldb * sizeof(float), ldb, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, jb, one, A + (i * lda + i) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
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

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasTrans, CBlasNoTrans, ib, jb, i, -one, A + i * lda * sizeof(float), lda, B + j * ldb * sizeof(float), ldb, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasLeft, CBlasUpper, CBlasTrans, diag, ib, jb, one, A + (i * lda + i) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
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

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, A + (i * lda + i + ib) * sizeof(float), lda, B + (j * ldb + i + ib) * sizeof(float), ldb, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasLeft, CBlasLower, CBlasTrans, diag, ib, jb, one, A + (i * lda + i) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
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

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasNoTrans, ib, jb, j, -one, B + i * sizeof(float), ldb, A + j * lda * sizeof(float), lda, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasRight, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, A + (j * lda + j) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
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

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasNoTrans, ib, jb, n - j - jb, -one, B + ((j + jb) * ldb + i) * sizeof(float), ldb, A + (j * lda + j + jb) * sizeof(float), lda, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasRight, CBlasLower, CBlasNoTrans, diag, ib, jb, one, A + (j * lda + j) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
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

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasTrans, ib, jb, n - j - jb, -one, B + ((j + jb) * ldb + i) * sizeof(float), ldb, A + ((j + jb) * lda + j) * sizeof(float), lda, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasRight, CBlasUpper, CBlasTrans, diag, ib, jb, one, A + (j * lda + j) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
          }
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuSgemm(sgemm, CBlasNoTrans, CBlasTrans, ib, jb, j, -one, B + i * sizeof(float), ldb, A + j * sizeof(float), lda, alpha, B + (j * ldb + i) * sizeof(float), ldb, stream));

            CU_ERROR_CHECK(cuStrsm2(module, CBlasRight, CBlasLower, CBlasTrans, diag, ib, jb, one, A + (j * lda + j) * sizeof(float), lda, B + (j * ldb + i) * sizeof(float), ldb, stream));
          }
        }
      }
    }

  }

  CU_ERROR_CHECK(cuModuleUnload(sgemm));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUStrsm(CUcontext * contexts, int deviceCount, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float alpha, const float * restrict A, size_t lda, float * restrict B, size_t ldb) {
  const size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 9;
  else if (ldb < max(1, m))
    info = 11;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0) return CUDA_SUCCESS;

  if (alpha == zero) {
    sgemm(CBlasNoTrans, CBlasNoTrans, m, n, 0, zero, A, lda, B, ldb, zero, B, ldb);
    return CUDA_SUCCESS;
  }

  const size_t mb = (side == CBlasLeft) ?  8 : 16;
  const size_t nb = (side == CBlasLeft) ? 16 :  8;

  if (m <= mb || n <= nb) {
    strsm(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[(i + ib) * lda + i], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasLeft, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
          }
        } while (i > 0);
      }
      else {
        for (size_t i = 0; i < m; i += mb) {
          const size_t ib = min(mb, m - i);

          for (size_t j = 0; j < n; j += nb) {
            const size_t jb = min(nb, n - j);

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, i, -one, &A[i], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasLeft, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasTrans, CBlasNoTrans, ib, jb, i, -one, &A[i * lda], lda, &B[j * ldb], ldb, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasLeft, CBlasUpper, CBlasTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasTrans, CBlasNoTrans, ib, jb, m - i - ib, -one, &A[i * lda + i + ib], lda, &B[j * ldb + i + ib], ldb, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasLeft, CBlasLower, CBlasTrans, diag, ib, jb, one, &A[i * lda + i], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, j, -one, &B[i], ldb, &A[j * lda], lda, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasRight, CBlasUpper, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasNoTrans, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[j * lda + j + jb], lda, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasRight, CBlasLower, CBlasNoTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
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

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasTrans, ib, jb, n - j - jb, -one, &B[(j + jb) * ldb + i], ldb, &A[(j + jb) * lda + j], lda, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasRight, CBlasUpper, CBlasTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        } while (j > 0);
      }
      else {
        for (size_t j = 0; j < n; j += nb) {
          const size_t jb = min(nb, n - j);

          for (size_t i = 0; i < m; i += mb) {
            const size_t ib = min(mb, m - i);

            CU_ERROR_CHECK(cuMultiGPUSgemm(contexts, deviceCount, CBlasNoTrans, CBlasTrans, ib, jb, j, -one, &B[i], ldb, &A[j], lda, alpha, &B[j * ldb + i], ldb));

            strsm(CBlasRight, CBlasLower, CBlasTrans, diag, ib, jb, one, &A[j * lda + j], lda, &B[j * ldb + i], ldb);
          }
        }
      }
    }
  }

  return CUDA_SUCCESS;
}

#if 0
// gcc -I../../include -I/opt/cuda/include -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -c strsm.c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

static void strsm_ref(CBlasSide, CBlasUplo, CBlasTranspose, CBlasDiag, size_t, size_t, float, const float * restrict, size_t, float * restrict, size_t);
static void * malloc2D(size_t, size_t, size_t *, size_t);
static void rand2D(size_t, size_t, float *, size_t);
static void fprintf2D(FILE *, const char *, size_t, size_t, const float *, size_t);
#ifdef GPU
static CUresult cuMemcpyHtoD2D(CUdeviceptr, size_t, size_t, size_t, const void *, size_t, size_t, size_t, size_t, size_t, size_t);
static CUresult cuMemcpyDtoH2D(void *, size_t, size_t, size_t, CUdeviceptr, size_t, size_t, size_t, size_t, size_t, size_t);
#endif

int main(int argc, char * argv[]) {
  CBlasSide side;
  CBlasUplo uplo;
  CBlasTranspose trans;
  CBlasDiag diag;
  size_t m, n;
#ifdef GPU
  int d;

  if (argc < 7 || argc > 8) {
    fprintf(stderr, "Usage: %s <side> <uplo> <trans> <diag> <m> <n> [device]\n"
    "where:\n"
    "  side     is 'l' or 'L' for CBlasLeft or 'r' or 'R' CBlasRight\n"
    "  uplo     is 'u' or 'U' for CBlasUpper or 'l' or 'L' CBlasLower\n"
    "  trans    is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
    "  diag     is 'n' or 'N' for CBlasNonUnit or 'u' or 'U' CBlasUnit\n"
    "  m and n  are the sizes of the matrices\n"
    "  device   is the ordinal of the GPU to use (default 0)\n", argv[0]);
    return 1;
  }
#else

  if (argc != 7) {
    fprintf(stderr, "Usage: %s <side> <uplo> <trans> <diag> <m> <n>\n"
    "where:\n"
    "  side     is 'l' or 'L' for CBlasLeft or 'r' or 'R' CBlasRight\n"
    "  uplo     is 'u' or 'U' for CBlasUpper or 'l' or 'L' CBlasLower\n"
    "  trans    is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
    "  diag     is 'n' or 'N' for CBlasNonUnit or 'u' or 'U' CBlasUnit\n"
    "  m and n  are the sizes of the matrices\n", argv[0]);
    return 1;
  }
#endif

  char s;
  if (sscanf(argv[1], "%c", &s) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (s) {
    case 'L': case 'l': side = CBlasLeft; break;
    case 'R': case 'r': side = CBlasRight; break;
    default: fprintf(stderr, "Unknown side '%c'\n", s); return 1;
  }

  char u;
  if (sscanf(argv[2], "%c", &u) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (u) {
    case 'U': case 'u': uplo = CBlasUpper; break;
    case 'L': case 'l': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 2;
  }

  char t;
  if (sscanf(argv[3], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[3]);
    return 3;
  }
  switch (t) {
    case 'N': case 'n': trans = CBlasNoTrans; break;
    case 'T': case 't': trans = CBlasTrans; break;
    case 'C': case 'c': trans = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 3;
  }

  char dg;
  if (sscanf(argv[4], "%c", &dg) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[4]);
    return 4;
  }
  switch (dg) {
    case 'N': case 'n': diag = CBlasNonUnit; break;
    case 'U': case 'u': diag = CBlasUnit; break;
    default: fprintf(stderr, "Unknown diag '%c'\n", dg); return 4;
  }

  if (sscanf(argv[5], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  if (sscanf(argv[6], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[6]);
    return 6;
  }

#ifdef GPU
  if (argc == 8) {
    if (sscanf(argv[7], "%d", &d) != 1) {
      fprintf(stderr, "Unable to parse number from '%s'\n", argv[7]);
      return 7;
    }
  }
  else
    d = 0;
#endif

  srand(0);

  float alpha, * A, * B, * refB;
  size_t lda, ldb;
#ifdef GPU
  CUdeviceptr dA, dB;
  size_t dlda, dldb;
#endif

#if defined(GPU) || defined(MULTIGPU)
  CU_ERROR_CHECK(cuInit(0));

#ifdef GPU
  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device));
#else
  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUcontext contexts[deviceCount];
  for (int i = 0; i < deviceCount; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));
    CU_ERROR_CHECK(cuCtxCreate(&contexts[i], CU_CTX_BLOCKING_SYNC, device));
  }
#endif
#endif

  rand2D(1, 1, &alpha, 0);
  if (m <= 8 && n <= 8)
    fprintf2D(stdout, "alpha", 1, 1, &alpha, 0);

  if (side == CBlasLeft) {
    if ((A = malloc2D(m, m, &lda, sizeof(float))) == NULL) {
      fprintf(stderr, "Unable to allocate A\n");
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    rand2D(m, m, A, lda);

#ifdef GPU
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, m * sizeof(float), m, sizeof(float)));
    dlda /= sizeof(float);
    CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, m, m, sizeof(float)));
#endif

    if (m <= 8 && n <= 8)
      fprintf2D(stdout, "A", m, m, A, lda);
  }
  else {
    if ((A = malloc2D(n, n, &lda, sizeof(float))) == NULL) {
      fprintf(stderr, "Unable to allocate A\n");
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    rand2D(n, n, A, lda);

#ifdef GPU
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(float), n, sizeof(float)));
    dlda /= sizeof(float);
    CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, n, n, sizeof(float)));
#endif

    if (m <= 8 && n <= 8)
      fprintf2D(stdout, "A", n, n, A, lda);
  }

  if ((B = malloc2D(m, n, &ldb, sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate B\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if ((refB = malloc2D(m, n, &ldb, sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate refB\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  rand2D(m, n, B, ldb);

  for (size_t j = 0; j < n; j++)
    memcpy(&refB[j * ldb], &B[j * ldb], m * sizeof(float));

#ifdef GPU
  CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, m * sizeof(float), n, sizeof(float)));
  dldb /= sizeof(float);
  CU_ERROR_CHECK(cuMemcpyHtoD2D(dB, dldb, 0, 0, B, ldb, 0, 0, m, n, sizeof(float)));
#endif

  if (m <= 8 && n <= 8)
    fprintf2D(stdout, "B", m, n, B, ldb);

  strsm_ref(side, uplo, trans, diag, m, n, alpha, A, lda, refB, ldb);
#ifdef GPU
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoad(&module, "strsm.cubin"));

  CU_ERROR_CHECK(cuStrsm(module, side, uplo, trans, diag, m, n, alpha, dA, dlda, dB, dldb, NULL));

  CU_ERROR_CHECK(cuMemcpyDtoH2D(B, ldb, 0, 0, dB, dldb, 0, 0, m, n, sizeof(float)));
#else
#ifdef MULTIGPU
  CU_ERROR_CHECK(cuMultiGPUStrsm(contexts, deviceCount, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
#else
  strsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#endif
#endif

  if (m <= 8 && n <= 8) {
    fprintf2D(stdout, "Reference STRSM", m, n, refB, ldb);
    fprintf2D(stdout, "STRSM", m, n, B, ldb);
  }

  bool passed = true;
  float diff = zero;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      float d = fabsf(B[j * ldb + i] - refB[j * ldb + i]);
      if (d > diff)
        diff = d;
      if (passed) {
        size_t flops;
        if (side == CBlasLeft) {
          if (uplo == CBlasUpper)
            flops = (trans == CBlasNoTrans) ? m - i - 1 : i;
          else
            flops = (trans == CBlasNoTrans) ? i : m - i - 1;
        }
        else {
          if (uplo == CBlasUpper)
            flops = (trans == CBlasNoTrans) ? j : n - j - 1;
          else
            flops = (trans == CBlasNoTrans) ? n - j - 1 : j;
        }
        if (alpha != one) flops += 1;
        if (diag == CBlasNonUnit) flops += 1;
        if (diff > (float)flops * 2.0f * FLT_EPSILON)
          passed = false;
      }
    }
  }

#ifdef GPU
  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuStrsm(module, side, uplo, trans, diag, m, n, alpha, dA, dlda, dB, dldb, NULL));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20000;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));
#else
  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }
  for (size_t i = 0; i < 20; i++)
#ifdef MULTIGPU
    CU_ERROR_CHECK(cuMultiGPUStrsm(contexts, deviceCount, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
#else
    strsm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
#endif
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
#endif

  size_t flops;
  if (alpha == zero)
    flops = 1;
  else {
    if (side == CBlasLeft)
      flops = m - 1;
    else
      flops = n - 1;
    if (alpha != one)
      flops += 1;
  }
  flops *= m * n;

  fprintf(stdout, "%.3ems %.3gGFlops/s Error: %.3e\n%sED!\n", time, ((float)flops * 1.e-9f) / time, diff, (passed) ? "PASS" : "FAIL");
//   fprintf(stdout, "%.3g\n", ((float)flops * 1.e-9f) / time);

  free(A);
  free(B);
  free(refB);
#ifdef GPU
  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dB));

#ifdef MULTIGPU
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuCtxDestroy(contexts[i]));
#else
  CU_ERROR_CHECK(cuModuleUnload(module));

  CU_ERROR_CHECK(cuCtxDestroy(context));
#endif
#endif

  return (int)!passed;
}

void strsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, float alpha, const float * A, size_t lda, float * B, size_t ldb) {
  size_t nRowA = (side == CBlasLeft) ? m : n;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 9;
  else if (ldb < max(1, m))
    info = 11;
  if (info != 0)
    XERBLA(info);

  if (m == 0 || n == 0) return;

  if (alpha == zero) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = zero;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = i + 1; k < m; k++)
              temp -= A[k * lda + i] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = 0; k < i; k++)
              temp -= A[k * lda + i] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = 0; k < i; k++)
              temp -= A[i * lda + k] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = i + 1; k < m; k++)
              temp -= A[i * lda + k] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
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
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = 0; k < j; k++)
              temp -= B[k * ldb + i] * A[j * lda + k];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        size_t j = n - 1;
        do {
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = j + 1; k < n; k++)
              temp -= B[k * ldb + i] * A[j * lda + k];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
    }
    else {
      if (uplo == CBlasUpper) {
        size_t j = n - 1;
        do {
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = j + 1; k < n; k++)
              temp -= B[k * ldb + i] * A[k * lda + j];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            float temp = B[j * ldb + i];
            if (alpha != one) temp *= alpha;
            for (size_t k = 0; k < j; k++)
              temp -= B[k * ldb + i] * A[k * lda + j];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
    }
  }
}

static float gaussian(float mean, float variance) {
  static float next;
  static bool hasNext = false;

  if (hasNext) {
    hasNext = false;
    return next * variance + mean;
  }

  float u0 = (float)(rand() + 1) / (float)RAND_MAX;
  float u1 = (float)(rand() + 1) / (float)RAND_MAX;
  float r = sqrtf(-1.0f * logf(u0));
  float phi = 2.0f * 3.1415926535f * u1;
  next = r * sinf(phi);
  hasNext = true;

  return r * cosf(phi) * variance + mean;
}

static inline void * malloc2D(size_t m, size_t n, size_t * ld, size_t elemSize) {
  size_t align = (16 / elemSize) - 1;
  *ld = (m + align) & ~align;
  return malloc(*ld * n * elemSize);
}

static void rand2D(size_t m, size_t n, float * A, size_t lda) {
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      A[j * lda + i] = gaussian(1.0f, 0.001f);
  }
}

static void fprintf2D(FILE * stream, const char * label, size_t m, size_t n, const float * A, size_t lda) {
  fprintf(stream, "%s =\n", label);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++)
      fprintf(stream, "%15.6f", A[j * lda + i]);
    fputs("\n", stream);
  }
}

#ifdef GPU
static CUresult cuMemcpyHtoD2D(CUdeviceptr A, size_t lda, size_t ai, size_t aj,
                               const void * B, size_t ldb, size_t bi, size_t bj,
                               size_t m, size_t n, size_t elemSize) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_HOST, B, 0, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_DEVICE, NULL, A, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2D(&copy);
}

static CUresult cuMemcpyDtoH2D(void * A, size_t lda, size_t ai, size_t aj,
                               CUdeviceptr B, size_t ldb, size_t bi, size_t bj,
                               size_t m, size_t n, size_t elemSize) {
  CUDA_MEMCPY2D copy = {
    bi * elemSize, bj, CU_MEMORYTYPE_DEVICE, NULL, B, 0, ldb * elemSize,
    ai * elemSize, aj, CU_MEMORYTYPE_HOST, A, 0, 0, lda * elemSize,
    m * elemSize, n };
  return cuMemcpy2D(&copy);
}
#endif
#endif
