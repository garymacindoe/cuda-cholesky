#include "blas.h"
#include "error.h"
#include <stdio.h>
#include "handle.h"
#include "config.h"
#include "dsyrk.fatbin.c"

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

void dsyrk(CBlasUplo uplo, CBlasTranspose trans,
           size_t n, size_t k,
           double alpha, const double * restrict A, size_t lda,
           double beta, double * restrict C, size_t ldc) {
  const size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < nRowA)
    info = 7;
  else if (ldc < n)
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one))
    return;

  if (alpha == zero) {
    if (uplo == CBlasUpper) {
      if (beta == zero) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    else {
      if (beta == zero) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    return;
  }

  if (trans == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        if (beta == zero) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = zero;
        }
        else if (beta != one) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] *= beta;
        }
        for (size_t l = 0; l < k; l++) {
          if (A[l * lda + j] != zero) {
            register double temp = alpha * A[l * lda + j];
            for (size_t i = 0; i <= j; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        if (beta == zero) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = zero;
        }
        else if (beta != one) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] *= beta;
        }
        for (size_t l = 0; l < k; l++) {
          if (A[l * lda + j] != zero) {
            register double temp = alpha * A[l * lda + j];
            for (size_t i = j; i < n; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i <= j; i++) {
          register double temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = j; i < n; i++) {
          register double temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
  }
}

CUresult cuDsyrk(CUblashandle handle, CBlasUplo uplo, CBlasTranspose trans,
                 size_t n, size_t k,
                 double alpha, CUdeviceptr A, size_t lda,
                 double beta, CUdeviceptr C, size_t ldc, CUstream stream) {
  const size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < nRowA)
    info = 7;
  else if (ldc < n)
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->dsyrk == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->dsyrk, imageBytes));

  const unsigned int mb = (trans == CBlasNoTrans) ? 64 : 32;
  const unsigned int nb = (trans == CBlasNoTrans) ?  8 : 16;
  const unsigned int kb = (trans == CBlasNoTrans && uplo == CBlasLower) ? 16 :  8;
  const unsigned int bx = 8;
  const unsigned int by = 8;

  char name[80];
  snprintf(name, 80,
           "_Z5dsyrkIL9CBlasUplo%dEL14CBlasTranspose%dELj%uELj%uELj%uELj%uELj%uEEvPKdPdddiiii",
           uplo, trans, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->dsyrk, name));

  void * params[] = { &A, &C, &alpha, &beta, &lda, &ldc, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(n + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  CU_ERROR_CHECK(cuCtxPopCurrent(&handle->context));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDsyrk(CUmultiGPUBlasHandle handle,
                         CBlasUplo uplo, CBlasTranspose trans,
                         size_t n, size_t k,
                         double alpha, const double * restrict A, size_t lda,
                         double beta, double * restrict C, size_t ldc) {
  const size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < nRowA)
    info = 7;
  else if (ldc < n)
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  if (alpha == zero) {
    if (uplo == CBlasUpper) {
      if (beta == zero) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    else {
      if (beta == zero) {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
#pragma omp parallel for
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    return CUDA_SUCCESS;
  }

  const size_t nb = (trans == CBlasNoTrans) ? DGEMM_N_MB : DGEMM_T_NB;

  if (n < nb) {
    dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  if (trans == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
      for (size_t j = nb; j < n; j += nb)
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle, CBlasNoTrans, CBlasTrans, j, min(n - j, nb), k, alpha, A, lda, &A[j], lda, beta, &C[j * ldc], ldc));
    }
    else {
      const size_t m = n - nb;
      for (size_t j = 0; j < m; j += nb) {
        const size_t jb = min(n - j, nb);
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle, CBlasNoTrans, CBlasTrans, n - j - jb, jb, k, alpha, &A[j + jb], lda, &A[j], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }

    for (size_t j = 0; j < n; j += nb)
      dsyrk(uplo, trans, min(n - j, nb), k, alpha, &A[j], lda, beta, &C[j * ldc + j], ldc);
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t j = nb; j < n; j += nb)
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle, CBlasTrans, CBlasNoTrans, j, min(n - j, nb), k, alpha, A, lda, &A[j * lda], lda, beta, &C[j * ldc], ldc));
    }
    else {
      const size_t m = n - nb;
      for (size_t j = 0; j < m; j += nb) {
        const size_t jb = min(n - j, nb);
        CU_ERROR_CHECK(cuMultiGPUDgemm(handle, CBlasTrans, CBlasNoTrans, n - j - jb, jb, k, alpha, &A[(j + jb) * lda], lda, &A[j * lda], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }

    for (size_t j = 0; j < n; j += nb)
      dsyrk(uplo, trans, min(n - j, nb), k, alpha, &A[j * lda], lda, beta, &C[j * ldc + j], ldc);
  }

  return CUDA_SUCCESS;
}
