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
static const double complex czero = 0.0 + 0.0 * I;

void zherk(CBlasUplo uplo, CBlasTranspose trans,
           size_t n, size_t k,
           double alpha, const double complex * restrict A, size_t lda,
           double beta, double complex * restrict C, size_t ldc) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (trans == CBlasTrans)
    info = 2;
  else if (lda < nRowA)
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
          for (size_t i = 0; i < j; i++)
            C[j * ldc + i] *= beta;
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
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
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
          for (size_t i = j + 1; i < n; i++)
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
          for (size_t i = 0; i < j; i++)
            C[j * ldc + i] *= beta;
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
        }
        else
          C[j * ldc + j] = creal(C[j * ldc + j]);
        for (size_t l = 0; l < k; l++) {
          if (A[l * lda + j] != zero) {
            register double complex temp = alpha * conj(A[l * lda + j]);
            for (size_t i = 0; i < j; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
            C[j * ldc + j] = creal(C[j * ldc + j]) + creal(temp * A[l * lda + j]);
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
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
          for (size_t i = j + 1; i < n; i++)
            C[j * ldc + i] *= beta;
        }
        else
          C[j * ldc + j] = creal(C[j * ldc + j]);
        for (size_t l = 0; l < k; l++) {
          if (A[l * lda + j] != zero) {
            register double complex temp = alpha * conj(A[l * lda + j]);
            C[j * ldc + j] = creal(C[j * ldc + j]) + creal(temp * A[l * lda + j]);
            for (size_t i = j + 1; i < n; i++)
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
        for (size_t i = 0; i < j; i++) {
          register double complex temp = czero;
          for (size_t l = 0; l < k; l++)
            temp += conj(A[i * lda + l]) * A[j * lda + l];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
        register double rtemp = zero;
        for (size_t l = 0; l < k; l++)
          rtemp += conj(A[j * lda + l]) * A[j * lda + l];
        if (beta == zero)
          C[j * ldc + j] = alpha * rtemp;
        else
          C[j * ldc + j] = alpha * rtemp + beta * creal(C[j * ldc + j]);
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        register double rtemp = zero;
        for (size_t l = 0; l < k; l++)
          rtemp += conj(A[j * lda + l]) * A[j * lda + l];
        if (beta == zero)
          C[j * ldc + j] = alpha * rtemp;
        else
          C[j * ldc + j] = alpha * rtemp + beta * creal(C[j * ldc + j]);
        for (size_t i = j + 1; i < n; i++) {
          register double complex temp = czero;
          for (size_t l = 0; l < k; l++)
            temp += conj(A[i * lda + l]) * A[j * lda + l];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
  }
}

CUresult cuZherk(CUmodule module,
                 CBlasUplo uplo, CBlasTranspose trans,
                 size_t n, size_t k,
                 double alpha, CUdeviceptr A, size_t lda,
                 double beta, CUdeviceptr C, size_t ldc, CUstream stream) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (trans == CBlasTrans)
    info = 2;
  else if (lda < nRowA)
    info = 7;
  else if (ldc < n)
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  const unsigned int mb = (trans == CBlasNoTrans) ? 64 :  8;
  const unsigned int nb = (trans == CBlasNoTrans) ?  4 :  8;
  const unsigned int kb = (trans == CBlasNoTrans) ? 16 :  4;
  const unsigned int bx = (trans == CBlasNoTrans) ?  4 :  4;
  const unsigned int by = (trans == CBlasNoTrans) ? 16 :  8;

  char name[71];
  snprintf(name, 71,
           "_Z6zherk%cIL9CBlasUplo%dELj%uELj%uELj%uELj%uELj%uEEvPK7double2PS1_ddiiii",
           trans, uplo, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &k, &alpha, &A, &lda, &beta, &C, &ldc };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(n + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}
#if 0
CUresult cuMultiGPUZherk(CBlasUplo uplo, CBlasTranspose trans,
                         size_t n, size_t k,
                         double alpha, const double complex * restrict A, size_t lda,
                         double beta, double complex * restrict C, size_t ldc) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (trans == CBlasTrans)
    info = 2;
  else if (lda < nRowA)
    info = 7;
  else if (ldc < n)
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return CUDA_SUCCESS;

  if (trans == CBlasNoTrans) {
    const size_t nb = 64;

    if (uplo == CBlasLower) {
      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        zherk(uplo, trans, jb, k, alpha, &A[j], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUZgemm(handles, deviceCount, trans, CBlasTrans, n - j - jb, jb, k, alpha, &A[j + jb], lda, &A[j], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }
    else {
      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        zherk(uplo, trans, jb, k, alpha, &A[j], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUZgemm(handles, deviceCount, trans, CBlasTrans, jb, n - j - jb, k, alpha, &A[j], lda, &A[j + jb], lda, beta, &C[(j + jb) * ldc + j], ldc));
      }
    }
  }
  else {
    const size_t nb = 64;

    if (uplo == CBlasLower) {
      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        zherk(uplo, trans, jb, k, alpha, &A[j * lda], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUZgemm(handles, deviceCount, trans, CBlasNoTrans, n - j - jb, jb, k, alpha, &A[(j + jb) * lda], lda, &A[j * lda], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }
    else {
      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        zherk(uplo, trans, jb, k, alpha, &A[j * lda], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUZgemm(handles, deviceCount, CBlasTrans, CBlasNoTrans, jb, n - j - jb, k, alpha, &A[j * lda], lda, &A[(j + jb) * lda], lda, beta, &C[(j + jb) * ldc + j], ldc));
      }
    }
  }

  return CUDA_SUCCESS;
}
#endif
