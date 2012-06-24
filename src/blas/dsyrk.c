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

static const double zero = 0.0;
static const double one = 1.0;

// #ifdef MKL_ILP64
//   extern void dsyrk_(const char *, const char *, const long *, const long *, const double *, const double *, const long *, const double *, double *, const long *);
// #else
//   extern void dsyrk_(const char *, const char *, const int *, const int *, const double *, const double *, const int *, const double *, double *, const int *);
// #endif
void dsyrk(CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k, double alpha, const double * restrict A, size_t lda, double beta, double * restrict C, size_t ldc) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
// #ifdef MKL_ILP64
//   dsyrk_((const char *)&uplo, (const char *)&trans, (const long *)&n, (const long *)&k, &alpha, A, (const long *)&lda, &beta, C, (const long *)&ldc);
// #else
//   dsyrk_((const char *)&uplo, (const char *)&trans, (const int *)&n, (const int *)&k, &alpha, A, (const int *)&lda, &beta, C, (const int *)&ldc);
// #endif
//   return;
  if (lda < max(1, nRowA))
    info = 7;
  else if (ldc < max(1, n))
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return;

  if (alpha == zero) {
    if (uplo == CBlasUpper) {
      if (beta == zero) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    else {
      if (beta == zero) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
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
            double temp = alpha * A[l * lda + j];
            for (size_t i = 0; i <= j; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else {
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
            double temp = alpha * A[l * lda + j];
            for (size_t i = j; i < n; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i <= j; i++) {
          double temp = zero;
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
      for (size_t j = 0; j < n; j++) {
        for (size_t i = j; i < n; i++) {
          double temp = zero;
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

static inline CUresult cuDsyrk2(CUmodule module, CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k, double alpha, CUdeviceptr A, size_t lda, double beta, CUdeviceptr C, size_t ldc, CUstream stream) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 7;
  else if (ldc < max(1, n))
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return CUDA_SUCCESS;

  const unsigned int mb = (trans == CBlasNoTrans) ? 64 : 32;
  const unsigned int nb = (trans == CBlasNoTrans) ? 16 : 32;
  const unsigned int kb = (trans == CBlasNoTrans) ? 16 :  8;
  const unsigned int bx = (trans == CBlasNoTrans) ? 16 :  8;
  const unsigned int by = (trans == CBlasNoTrans) ?  4 :  8;

  char name[82];
  snprintf(name, 82, "_Z5ssyrkIL9CBlasUplo%dEL14CBlasTranspose%dELj%uELj%uELj%uELj%uELj%uEEviifPKfifPfi", uplo, trans, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &n, &k, &alpha, &A, &lda, &beta, &C, &ldc };

  CU_ERROR_CHECK(cuLaunchKernel(function, maxj(1, ((unsigned int)n + mb - 1) / mb), maxj(1, ((unsigned int)n + nb - 1) / nb), 1, bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

CUresult cuDsyrk(CUmodule module, CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k, double alpha, CUdeviceptr A, size_t lda, double beta, CUdeviceptr C, size_t ldc, CUstream compute) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 7;
  else if (ldc < max(1, n))
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return CUDA_SUCCESS;

  double * hA0, *hA1, * hC;
  size_t hlda0, hlda1, hldc;
  CUstream copy0, copy1;
  CUmodule dgemm;

  if (trans == CBlasNoTrans) {
    if (uplo == CBlasLower) {
      const size_t nb = 64;
      const size_t kb = 256;

      if (n < nb && k < kb)
        return cuDsyrk2(module, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, compute);

      CU_ERROR_CHECK(cuModuleLoad(&dgemm, "dgemm.cubin"));

      CU_ERROR_CHECK(cuStreamCreate(&copy0, 0));
      CU_ERROR_CHECK(cuStreamCreate(&copy1, 0));

      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA0, (hlda0 = (nb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA1, (hlda1 = (nb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hC,  (hldc  = (nb + 1u) & ~1u) * nb * sizeof(double)));

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);

        if (j + jb < n)
          CU_ERROR_CHECK(cuDgemm(dgemm, trans, CBlasTrans, n - j - jb, jb, k, alpha, A + (j + jb) * sizeof(double), lda, A + j * sizeof(double), lda, beta, C + (j * ldc + j + jb) * sizeof(double), ldc, compute));

        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hC,  hldc,  0, 0, C, ldc, j, j, jb, jb, sizeof(double), copy1));
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, j, 0, jb, min(kb, k), sizeof(double), copy0));

        CU_ERROR_CHECK(cuStreamSynchronize(copy1));
        dsyrk(uplo, trans, jb, 0, zero, NULL, lda, beta, hC, hldc);

        for (size_t l = 0; l < k; l += kb) {
          if (l + kb < k)
            CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA1, hlda1, 0, 0, A, lda, j, l + kb, jb, min(kb, k - l - kb), sizeof(double), copy1));

          CU_ERROR_CHECK(cuStreamSynchronize(copy0));
          dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA0, hlda0, one, hC, hldc);

          l += kb;
          if (l < k) {
            if (l + kb < k)
              CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, j, l + kb, jb, min(kb, k - l - kb), sizeof(double), copy0));

            CU_ERROR_CHECK(cuStreamSynchronize(copy1));
            dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA1, hlda1, one, hC, hldc);
          }
        }

        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, j, j, hC, hldc, 0, 0, jb, jb, sizeof(double), copy0));
      }
    }
    else {
      const size_t nb = 64;
      const size_t kb = 256;

      if (n < nb && k < kb)
        return cuDsyrk2(module, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, compute);

      CU_ERROR_CHECK(cuModuleLoad(&dgemm, "dgemm.cubin"));

      CU_ERROR_CHECK(cuStreamCreate(&copy0, 0));
      CU_ERROR_CHECK(cuStreamCreate(&copy1, 0));

      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA0, (hlda0 = (nb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA1, (hlda1 = (nb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hC,  (hldc  = (nb + 1u) & ~1u) * nb * sizeof(double)));

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);

        if (j + jb < n)
          CU_ERROR_CHECK(cuDgemm(dgemm, trans, CBlasTrans, jb, n - j - jb, k, alpha, A + j * sizeof(double), lda, A + (j + jb) * sizeof(double), lda, beta, C + ((j + jb) * ldc + j) * sizeof(double), ldc, compute));

        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hC,  hldc,  0, 0, C, ldc, j, j, jb, jb, sizeof(double), copy1));
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, j, 0, jb, min(kb, k), sizeof(double), copy0));

        CU_ERROR_CHECK(cuStreamSynchronize(copy1));
        dsyrk(uplo, trans, jb, 0, zero, NULL, lda, beta, hC, hldc);

        for (size_t l = 0; l < k; l += kb) {
          if (l + kb < k)
            CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA1, hlda1, 0, 0, A, lda, j, l + kb, jb, min(kb, k - l - kb), sizeof(double), copy1));

          CU_ERROR_CHECK(cuStreamSynchronize(copy0));
          dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA0, hlda0, one, hC, hldc);

          l += kb;
          if (l < k) {
            if (l + kb < k)
              CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, j, l + kb, jb, min(kb, k - l - kb), sizeof(double), copy0));

            CU_ERROR_CHECK(cuStreamSynchronize(copy1));
            dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA1, hlda1, one, hC, hldc);
          }
        }

        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, j, j, hC, hldc, 0, 0, jb, jb, sizeof(double), copy0));
      }
    }
  }
  else {
    if (uplo == CBlasLower) {
      const size_t nb = 64;
      const size_t kb = 256;

      if (n < nb && k < kb)
        return cuDsyrk2(module, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, compute);

      CU_ERROR_CHECK(cuModuleLoad(&dgemm, "dgemm.cubin"));

      CU_ERROR_CHECK(cuStreamCreate(&copy0, 0));
      CU_ERROR_CHECK(cuStreamCreate(&copy1, 0));

      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA0, (hlda0 = (kb + 1u) & ~1u) * nb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA1, (hlda1 = (kb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hC,  (hldc  = (nb + 1u) & ~1u) * nb * sizeof(double)));

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);

        if (j + jb < n)
          CU_ERROR_CHECK(cuDgemm(dgemm, trans, CBlasNoTrans, n - j - jb, jb, k, alpha, A + (j + jb) * lda * sizeof(double), lda, A + j * lda * sizeof(double), lda, beta, C + (j * ldc + j + jb) * sizeof(double), ldc, compute));

        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hC,  hldc,  0, 0, C, ldc, j, j, jb, jb, sizeof(double), copy1));
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, 0, j, min(kb, k), jb, sizeof(double), copy0));

        CU_ERROR_CHECK(cuStreamSynchronize(copy1));
        dsyrk(uplo, trans, jb, 0, zero, NULL, lda, beta, hC, hldc);

        for (size_t l = 0; l < k; l += kb) {
          if (l + kb < k)
            CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA1, hlda1, 0, 0, A, lda, l + kb, j, min(kb, k - l - kb), jb, sizeof(double), copy1));

          CU_ERROR_CHECK(cuStreamSynchronize(copy0));
          dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA0, hlda0, one, hC, hldc);

          l += kb;
          if (l < k) {
            if (l + kb < k)
              CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, l + kb, j, min(kb, k - l - kb), jb, sizeof(double), copy0));

            CU_ERROR_CHECK(cuStreamSynchronize(copy1));
            dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA1, hlda1, one, hC, hldc);
          }
        }

        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, j, j, hC, hldc, 0, 0, jb, jb, sizeof(double), copy0));
      }
    }
    else {
      const size_t nb = 64;
      const size_t kb = 256;

      if (n < nb && k < kb)
        return cuDsyrk2(module, uplo, trans, n, k, alpha, A, lda, beta, C, ldc, compute);

      CU_ERROR_CHECK(cuModuleLoad(&dgemm, "dgemm.cubin"));

      CU_ERROR_CHECK(cuStreamCreate(&copy0, 0));
      CU_ERROR_CHECK(cuStreamCreate(&copy1, 0));

      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA0, (hlda0 = (kb + 1u) & ~1u) * nb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hA1, (hlda1 = (kb + 1u) & ~1u) * kb * sizeof(double)));
      CU_ERROR_CHECK(cuMemAllocHost((void **)&hC,  (hldc  = (nb + 1u) & ~1u) * nb * sizeof(double)));

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);

        if (j + jb < n)
          CU_ERROR_CHECK(cuDgemm(dgemm, CBlasTrans, CBlasNoTrans, jb, n - j - jb, k, alpha, A + j * lda * sizeof(double), lda, A + (j + jb) * lda * sizeof(double), lda, beta, C + ((j + jb) * ldc + j) * sizeof(double), ldc, compute));

        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hC,  hldc,  0, 0, C, ldc, j, j, jb, jb, sizeof(double), copy1));
        CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, 0, j, min(kb, k), jb, sizeof(double), copy0));

        CU_ERROR_CHECK(cuStreamSynchronize(copy1));
        dsyrk(uplo, trans, jb, 0, zero, NULL, lda, beta, hC, hldc);

        for (size_t l = 0; l < k; l += kb) {
          if (l + kb < k)
            CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA1, hlda1, 0, 0, A, lda, l + kb, j, min(kb, k - l - kb), jb, sizeof(double), copy1));

          CU_ERROR_CHECK(cuStreamSynchronize(copy0));
          dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA0, hlda0, one, hC, hldc);

          l += kb;
          if (l < k) {
            if (l + kb < k)
              CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(hA0, hlda0, 0, 0, A, lda, l + kb, j, min(kb, k - l - kb), jb, sizeof(double), copy0));

            CU_ERROR_CHECK(cuStreamSynchronize(copy1));
            dsyrk(uplo, trans, jb, min(kb, k - l), alpha, hA1, hlda1, one, hC, hldc);
          }
        }

        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, j, j, hC, hldc, 0, 0, jb, jb, sizeof(double), copy0));
      }
    }
  }

  CU_ERROR_CHECK(cuModuleUnload(dgemm));

  CU_ERROR_CHECK(cuMemFreeHost(hA0));
  CU_ERROR_CHECK(cuMemFreeHost(hA1));
  CU_ERROR_CHECK(cuMemFreeHost(hC));

  CU_ERROR_CHECK(cuStreamDestroy(copy0));
  CU_ERROR_CHECK(cuStreamDestroy(copy1));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDsyrk(CUcontext * contexts, int deviceCount, CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k, double alpha, const double * restrict A, size_t lda, double beta, double * restrict C, size_t ldc) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

  int info = 0;
  if (lda < max(1, nRowA))
    info = 7;
  else if (ldc < max(1, n))
    info = 10;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return CUDA_SUCCESS;

  if (trans == CBlasNoTrans) {
    if (uplo == CBlasLower) {
      const size_t nb = 64;

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        dsyrk(uplo, trans, jb, k, alpha, &A[j], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, trans, CBlasTrans, n - j - jb, jb, k, alpha, &A[j + jb], lda, &A[j], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }
    else {
      const size_t nb = 64;

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        dsyrk(uplo, trans, jb, k, alpha, &A[j], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, trans, CBlasTrans, jb, n - j - jb, k, alpha, &A[j], lda, &A[j + jb], lda, beta, &C[(j + jb) * ldc + j], ldc));
      }
    }
  }
  else {
    if (uplo == CBlasLower) {
      const size_t nb = 64;

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        dsyrk(uplo, trans, jb, k, alpha, &A[j * lda], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, trans, CBlasNoTrans, n - j - jb, jb, k, alpha, &A[(j + jb) * lda], lda, &A[j * lda], lda, beta, &C[j * ldc + j + jb], ldc));
      }
    }
    else {
      const size_t nb = 64;

      for (size_t j = 0; j < n; j += nb) {
        const size_t jb = min(nb, n - j);
        dsyrk(uplo, trans, jb, k, alpha, &A[j * lda], lda, beta, &C[j * ldc + j], ldc);
        if (j + jb < n)
          CU_ERROR_CHECK(cuMultiGPUDgemm(contexts, deviceCount, CBlasTrans, CBlasNoTrans, jb, n - j - jb, k, alpha, &A[j * lda], lda, &A[(j + jb) * lda], lda, beta, &C[(j + jb) * ldc + j], ldc));
      }
    }
  }

  return CUDA_SUCCESS;
}

#if 0
// gcc -I../../include -I/opt/cuda/include -march=native -O2 -pipe -std=c99 -pedantic -Wall -Wextra -Wconversion -c dsyrk.c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

static void dsyrk_ref(CBlasUplo, CBlasTranspose, size_t, size_t, double, const double * restrict, size_t, double, double * restrict, size_t);
static void * malloc2D(size_t, size_t, size_t *, size_t);
static void rand2D(size_t, size_t, double *, size_t);
static void fprintf2D(FILE *, const char *, size_t, size_t, const double *, size_t);
#ifdef GPU
static CUresult cuMemcpyHtoD2D(CUdeviceptr, size_t, size_t, size_t, const void *, size_t, size_t, size_t, size_t, size_t, size_t);
static CUresult cuMemcpyDtoH2D(void *, size_t, size_t, size_t, CUdeviceptr, size_t, size_t, size_t, size_t, size_t, size_t);
#endif

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  CBlasTranspose trans;
  size_t n, k;
#ifdef GPU
  int d;

  if (argc < 5 || argc > 6) {
    fprintf(stderr, "Usage: %s <uplo> <trans> <n> <k> [device]\nwhere:\n  uplo is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n  trans   is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n  n and k            are the sizes of the matrices\n  device             is the ordinal of the GPU to use (default 0)\n", argv[0]);
    return 1;
  }
#else

  if (argc != 5) {
    fprintf(stderr, "Usage: %s <uplo> <trans> <n> <k>\nwhere:\n  uplo is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n  trans   is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n  n and k            are the sizes of the matrices\n", argv[0]);
    return 1;
  }
#endif

  char u;
  if (sscanf(argv[1], "%c", &u) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (u) {
    case 'U': case 'u': uplo = CBlasUpper; break;
    case 'L': case 'l': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 1;
  }

  char t;
  if (sscanf(argv[2], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (t) {
    case 'N': case 'n': trans = CBlasNoTrans; break;
    case 'T': case 't': trans = CBlasTrans; break;
    case 'C': case 'c': trans = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[3], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[4]);
    return 3;
  }

  if (sscanf(argv[4], "%zu", &k) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 4;
  }
#ifdef GPU
  if (argc == 6) {
    if (sscanf(argv[5], "%d", &d) != 1) {
      fprintf(stderr, "Unable to parse number from '%s'\n", argv[6]);
      return 5;
    }
  }
  else
    d = 0;
#endif
  srand(0);

  double alpha, beta, * A, * C, * refC;
  size_t lda, ldc;
#ifdef GPU
  CUdeviceptr dA, dC;
  size_t dlda, dldc;
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
  rand2D(1, 1, &beta, 0);
  if (n <= 8 && k <= 8) {
    fprintf2D(stdout, "alpha", 1, 1, &alpha, 0);
    fprintf2D(stdout, "beta", 1, 1, &beta, 0);
  }

  if (trans == CBlasNoTrans) {
    if ((A = malloc2D(n, k, &lda, sizeof(double))) == NULL) {
      fprintf(stderr, "Unable to allocate A\n");
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    rand2D(n, k, A, lda);

#ifdef GPU
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(double), k, sizeof(double)));
    dlda /= sizeof(double);
    CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, n, k, sizeof(double)));
#endif

    if (n <= 8 && k <= 8)
      fprintf2D(stdout, "A", n, k, A, lda);
  }
  else {
    if ((A = malloc2D(k, n, &lda, sizeof(double))) == NULL) {
      fprintf(stderr, "Unable to allocate A\n");
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    rand2D(k, n, A, lda);

#ifdef GPU
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, k * sizeof(double), n, sizeof(double)));
    dlda /= sizeof(double);
    CU_ERROR_CHECK(cuMemcpyHtoD2D(dA, dlda, 0, 0, A, lda, 0, 0, k, n, sizeof(double)));
#endif

    if (n <= 8 && k <= 8)
      fprintf2D(stdout, "A", k, n, A, lda);
  }

  if ((C = malloc2D(n, n, &ldc, sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate C\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if ((refC = malloc2D(n, n, &ldc, sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate refC\n");
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  rand2D(n, n, C, ldc);

  for (size_t j = 0; j < n; j++)
    memcpy(&refC[j * ldc], &C[j * ldc], n * sizeof(double));

#ifdef GPU
  CU_ERROR_CHECK(cuMemAllocPitch(&dC, &dldc, n * sizeof(double), n, sizeof(double)));
  dldc /= sizeof(double);
  CU_ERROR_CHECK(cuMemcpyHtoD2D(dC, dldc, 0, 0, C, ldc, 0, 0, n, n, sizeof(double)));
#endif

  if (n <= 8 && k <= 8)
    fprintf2D(stdout, "C", n, n, C, ldc);

  dsyrk_ref(uplo, trans, n, k, alpha, A, lda, beta, refC, ldc);
#ifdef GPU
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoad(&module, "dsyrk.cubin"));

  CU_ERROR_CHECK(cuDsyrk(module, uplo, trans, n, k, alpha, dA, dlda, beta, dC, dldc, NULL));

  CU_ERROR_CHECK(cuMemcpyDtoH2D(C, ldc, 0, 0, dC, dldc, 0, 0, n, n, sizeof(double)));
#else
#ifdef MULTIGPU
  CU_ERROR_CHECK(cuMultiGPUDsyrk(contexts, deviceCount, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
#else
  dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
#endif

  if (n <= 8 && k <= 8) {
    fprintf2D(stdout, "Reference DSYRK", n, n, refC, ldc);
    fprintf2D(stdout, "DSYRK", n, n, C, ldc);
  }

  double diff = zero;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      double d = fabs(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }

#ifdef GPU
  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuDsyrk(module, uplo, trans, n, k, alpha, dA, dlda, beta, dC, dldc, NULL));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  double time;
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
    CU_ERROR_CHECK(cuMultiGPUDsyrk(contexts, deviceCount, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
#else
    dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
#endif

  size_t flops = 2 * k - 1;
  if (alpha != one)
    flops += 1;
  if (beta != zero)
    flops += 2;
  double error = (double)flops * 2.0f * FLT_EPSILON;
  flops *= (n * (n + 1)) / 2;

  bool passed = (diff <= error);
  fprintf(stdout, "%.3ems %.3gGFlops/s Error: %.3e\n%sED!\n", time, ((double)flops * 1.e-9f) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(C);
  free(refC);
#ifdef GPU
  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dC));

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

static void dsyrk_ref(CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k, double alpha, const double * restrict A, size_t lda, double beta, double * restrict C, size_t ldc) {
  if (n == 0 || ((k == 0 || alpha == zero) && beta == one)) return;

  if (alpha == zero) {
    if (beta == zero) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = zero;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = zero;
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] *= beta;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] *= beta;
        }
      }
    }
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i <= j; i++) {

        double temp;
        if (trans == CBlasNoTrans) {
          temp = A[i] * A[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * A[l * lda + j];
        }
        else {
          temp = A[i * lda] * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
        }

        if (alpha != one)
          temp *= alpha;
        if (beta != zero)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;

      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {

        double temp;
        if (trans == CBlasNoTrans) {
          temp = A[i] * A[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * A[l * lda + j];
        }
        else {
          temp = A[i * lda] * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
        }

        if (alpha != one)
          temp *= alpha;
        if (beta != zero)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;

      }
    }
  }
}

static void * malloc2D(size_t m, size_t n, size_t * ld, size_t elemSize) {
  size_t align = (16 / elemSize) - 1;
  *ld = (m + align) & ~align;
  return malloc(*ld * n * elemSize);
}

static void rand2D(size_t m, size_t n, double * A, size_t lda) {
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      A[j * lda + i] = (double)rand() / (double)RAND_MAX;
  }
}

static void fprintf2D(FILE * stream, const char * label, size_t m, size_t n, const double * A, size_t lda) {
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
