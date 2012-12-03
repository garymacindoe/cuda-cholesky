#include "blas.h"
#include "error.h"
#include "../multigpu.h"
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

static const double complex zero = 0.0 + 0.0 * I;
static const double complex one = 1.0 + 0.0 * I;

void zgemm(CBlasTranspose transA, CBlasTranspose transB,
           size_t m, size_t n, size_t k,
           double complex alpha, const double complex * restrict A, size_t lda, const double complex * restrict B, size_t ldb,
           double complex beta, double complex * restrict C, size_t ldc) {
  size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  size_t nRowB = (transB == CBlasNoTrans) ? k : n;

  int info = 0;
  if (lda < nRowA)
    info = 8;
  else if (ldb < nRowB)
    info = 10;
  else if (ldc < m)
    info = 13;
  if (info != 0) {
    XERBLA(info);
    return;
  }

  if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
    return;

  if (alpha == zero) {
    if (beta == zero) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = zero;
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] *= beta;
      }
    }
    return;
  }

  if (transB == CBlasNoTrans) {
    if (transA == CBlasNoTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        if (beta == zero) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] = zero;
        }
        else if (beta != one) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] *= beta;
        }
        for (size_t l = 0; l < k; l++) {
          if (B[j * ldb + l] != zero) {
            register double complex temp = alpha * B[j * ldb + l];
            for (size_t i = 0; i < m; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else if (transA == CBlasConjTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conj(A[i * lda + l]) * B[j * ldb + l];
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
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * B[j * ldb + l];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
  }
  else if (transB == CBlasConjTrans) {
    if (transA == CBlasNoTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        if (beta == zero) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] = zero;
        }
        else if (beta != one) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] *= beta;
        }
        for (size_t l = 0; l < k; l++) {
          if (B[l * ldb + j] != zero) {
            register double complex temp = alpha * conj(B[l * ldb + j]);
            for (size_t i = 0; i < m; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else if (transA == CBlasConjTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conj(A[i * lda + l]) * conj(B[l * ldb + j]);
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
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * conj(B[l * ldb + j]);
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        if (beta == zero) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] = zero;
        }
        else if (beta != one) {
          for (size_t i = 0; i < m; i++)
            C[j * ldc + i] *= beta;
        }
        for (size_t l = 0; l < k; l++) {
          if (B[l * ldb + j] != zero) {
            register double complex temp = alpha * B[l * ldb + j];
            for (size_t i = 0; i < m; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else if (transA == CBlasConjTrans) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conj(A[i * lda + l]) * B[l * ldb + j];
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
        for (size_t i = 0; i < m; i++) {
          register double complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * B[l * ldb + j];
          if (beta == zero)
            C[j * ldc + i] = alpha * temp;
          else
            C[j * ldc + i] = alpha * temp + beta * C[j * ldc + i];
        }
      }
    }
  }
}

CUresult cuZgemm2(CUmodule module,
                  CBlasTranspose transA, CBlasTranspose transB,
                  size_t m, size_t n, size_t k,
                  double complex alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
                  double complex beta, CUdeviceptr C, size_t ldc, CUdeviceptr D, size_t ldd,
                  CUstream stream) {
  size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  size_t nRowB = (transB == CBlasNoTrans) ? k : n;

  int info = 0;
  if (lda < nRowA)
    info = 8;
  else if (ldb < nRowB)
    info = 10;
  else if (ldc < m)
    info = 13;
  else if (ldd < m)
    info = 15;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  unsigned int mb, nb, kb, bx, by;
  char name[95];

  if (transA == CBlasNoTrans) {
    mb = 64; nb =  4; kb = 16;
    bx = (transB == CBlasNoTrans) ? 16 :  4;
    by = (transB == CBlasNoTrans) ?  4 : 16;
    snprintf(name, 90, "_Z6zgemmNIL14CBlasTranspose%dELj64ELj4ELj16ELj%uELj%uEEvPK7double2S3_S3_PS1_S1_S1_iiiiiii", transB, bx, by);
  }
  else {
    mb =  8;
    nb = (transB == CBlasNoTrans) ?  8 : 16;
    kb = (transB == CBlasNoTrans) ?  4 :  8;
    bx = (transB == CBlasNoTrans) ?  4 :  8;
    by =  8;
    snprintf(name, 95, "_Z6zgemmTIL14CBlasTranspose%dELS0_%dELj8ELj%uELj%uELj%uELj8EEvPK7double2S3_S3_PS1_S1_S1_iiiiiii", transA, transB, nb, kb, bx);
  }

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &A, &B, &C, &D, &alpha, &beta, &lda, &ldb, &ldc, &ldd, &m, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

struct zgemm_args {
  CBlasTranspose transA, transB;
  size_t m, n, k;
  double complex alpha; const double complex * A; size_t lda; const double complex * B; size_t ldb;
  double complex beta; double complex * C; size_t ldc;
};

static CUresult background_zgemm(const void * a) {
  struct zgemm_args * args = (struct zgemm_args *)a;

  const CBlasTranspose transA = args->transA;
  const CBlasTranspose transB = args->transB;
  const size_t m = args->m;
  const size_t n = args->n;
  const size_t k = args->k;
  const double complex alpha = args->alpha;
  const double complex beta = args->beta;

  if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc;

  const size_t kb = (transA == CBlasNoTrans) ? 256 : 264;

  // Load the cgemm module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoad(&module, "zgemm.fatbin"));

  // Create separate streams for concurrent copy and execute
  CUstream compute, copy;
  CU_ERROR_CHECK(cuStreamCreate(&compute, 0));
  CU_ERROR_CHECK(cuStreamCreate(&copy, 0));

  // Allocate C
  CU_ERROR_CHECK(cuMemAllocPitch(&C, &ldc, m * sizeof(double complex), n, sizeof(double complex)));
  ldc /= sizeof(double complex);

  // Copy C onto the device
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, 0, 0,
                                     args->C, args->ldc, 0, 0,
                                     m, n, sizeof(double complex), compute));

  // Perform C *= beta
  CU_ERROR_CHECK(cuZgemm(module, CBlasNoTrans, CBlasNoTrans,
                         m, n, 0,
                         zero, 0, ldc, 0, 0, beta, C, ldc, compute));

  // Can exit early if alpha * op(A) * op(B) will evaluate to zero
  if (alpha != zero && k > 0) {
    // Perform C += alpha * op(A) * op(B)
    if (transB == CBlasNoTrans) {
      // B is k * n
      CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, kb * sizeof(double complex), n, sizeof(double complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, kb * sizeof(double complex), n, sizeof(double complex)));
      ldb /= sizeof(double complex);

      if (transA == CBlasNoTrans) {
        // A is m * k
        CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, m * sizeof(double complex), kb, sizeof(double complex)));
        CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, m * sizeof(double complex), kb, sizeof(double complex)));
        lda /= sizeof(double complex);

        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                          args->A, args->lda, 0, 0,
                                          m, lb, sizeof(double complex), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                          args->B, args->ldb, 0, 0,
                                          lb, n, sizeof(double complex), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuZgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, lda, B0, ldb, one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, 0, l + kb,
                                               m, lb, sizeof(double complex), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, l + kb, 0,
                                               lb, n, sizeof(double complex), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // A is k * m
        CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, kb * sizeof(double complex), m, sizeof(double complex)));
        CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, kb * sizeof(double complex), m, sizeof(double complex)));
        lda /= sizeof(double complex);

        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                          args->A, args->lda, 0, 0,
                                          lb, m, sizeof(double complex), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                          args->B, args->ldb, 0, 0,
                                          lb, n, sizeof(double complex), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuZgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, lda, B0, ldb, one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, l + kb, 0,
                                               lb, m, sizeof(double complex), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, l + kb, 0,
                                               lb, n, sizeof(double complex), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
    }
    else {
      // B is n * k
      CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, n * sizeof(double complex), kb, sizeof(double complex)));
      CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, n * sizeof(double complex), kb, sizeof(double complex)));
      ldb /= sizeof(double complex);

      if (transA == CBlasNoTrans) {
        // A is m * k
        CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, m * sizeof(double complex), kb, sizeof(double complex)));
        CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, m * sizeof(double complex), kb, sizeof(double complex)));
        lda /= sizeof(double complex);

        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           m, lb, sizeof(double complex), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           n, lb, sizeof(double complex), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuZgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, lda, B0, ldb, one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                              args->A, args->lda, 0, l + kb,
                                              m, lb, sizeof(double complex), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                              args->B, args->ldb, 0, l + kb,
                                              n, lb, sizeof(double complex), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // A is k * m
        CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, kb * sizeof(double complex), m, sizeof(double complex)));
        CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, kb * sizeof(double complex), m, sizeof(double complex)));
        lda /= sizeof(double complex);

        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                          args->A, args->lda, 0, 0,
                                          lb, m, sizeof(double complex), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                          args->B, args->ldb, 0, 0,
                                          n, lb, sizeof(double complex), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuZgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, lda, B0, ldb, one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, l + kb, 0,
                                               lb, m, sizeof(double complex), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, 0, l + kb,
                                               n, lb, sizeof(double complex), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
    }
  }

  // Copy C back onto the host on the compute stream
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(args->C, args->ldc, 0, 0,
                                     C, ldc, 0, 0,
                                     m, n, sizeof(double complex), compute));

  // Free A, B and C
  CU_ERROR_CHECK(cuMemFree(A0));
  CU_ERROR_CHECK(cuMemFree(A1));
  CU_ERROR_CHECK(cuMemFree(B0));
  CU_ERROR_CHECK(cuMemFree(B1));
  CU_ERROR_CHECK(cuMemFree(C));

  // Destroy the streams
  CU_ERROR_CHECK(cuStreamDestroy(compute));
  CU_ERROR_CHECK(cuStreamDestroy(copy));

  // Unload the module
  CU_ERROR_CHECK(cuModuleUnload(module));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUZgemm(CUmultiGPU multiGPU,
                         CBlasTranspose transA, CBlasTranspose transB,
                         size_t m, size_t n, size_t k,
                         double complex alpha, const double complex * restrict A, size_t lda,
                         const double complex * restrict B, size_t ldb,
                         double complex beta, double complex * restrict C, size_t ldc) {
  size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  size_t nRowB = (transB == CBlasNoTrans) ? k : n;

  int info = 0;
  if (lda < nRowA)
    info = 8;
  else if (ldb < nRowB)
    info = 10;
  else if (ldc < m)
    info = 13;
  if (info != 0) {
    XERBLA(info);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  if (alpha == zero) {
    if (beta == zero) {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = zero;
      }
    }
    else {
      for (size_t j = 0; j < n; j++) {
#pragma omp parallel for
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] *= beta;
      }
    }
    return CUDA_SUCCESS;
  }

  /**
   * When transA == CBlasNoTrans each GPU MP processes blocks of 64x4 using 64
   * threads per block.
   * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
   * to mask memory latency (64 * 3 = 192 threads/6 warps).
   * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 10x24, 12x20,
   * 15x16, etc. block size here.
   * 10x24 is chosen to retain the m >> n behaviour needed for ZPOTRF('L',..).
   * mb = 10 * 64 = 640
   * nb = 24 *  4 =  96
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  kb >= 256 gives ~79GFlops/s.  This requires (640 * 96 + 2 *
   * 256 * (640 + 96)) * 16 = 6848kB of graphics memory.
   *
   * These block sizes give a bandwidth reduction of 2 / (1/640 + 1/96) = 166.96
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA == CBlasNoTrans is
   * (79 * 10^9) / (6 * 1,073,741,824 / sizeof(double complex)) = 196.20
   *
   * When transA != CBlasNoTrans and transB == CBlasNoTrans each GPU MP processes
   * blocks of 8x8 using 32 threads per block.
   * There are 30 MPs on the GTX 280 and each requires a minimum of 6 blocks
   * to mask memory latency (32 * 6 = 192 threads/6 warps).
   * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 10x24, 12x20,
   * 15x16, etc. block size here.
   * 8x15 is chosen to retain the m << n behaviour needed for CPOTRF('U',..).
   * mb =  4 * 32 = 128
   * nb = 30 * 16 = 480
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  264 <= kb <= 480 gives ~305GFlops/s.  This requires (128 * 480
   * + 2 * 264 * (128 + 480)) * 8 = 2984kB of graphics memory.
   *
   * These block sizes give a bandwidth reduction of 2 / (1/128 + 1/480) = 202.11
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA != CBlasNoTrans is
   * (305 * 10^9) / (6 * 1,073,741,824 / sizeof(float complex)) = 378.74
   *
   */
  const size_t mb = (transA == CBlasNoTrans) ? 640 : 128;
  const size_t nb = (transA == CBlasNoTrans) ?  96 : 480;

  if (m < mb && n < nb) {
    zgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  const size_t nTasks = ((m + mb - 1) / mb) * ((n + nb - 1) / nb);
  CUtask * tasks;
  if ((tasks = malloc(nTasks * sizeof(CUtask))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  size_t t = 0;

  struct zgemm_args args = { transA, transB,
                             m, n, k,
                             alpha, A, lda, B, ldb,
                             beta, C, ldc };

  if (transB == CBlasNoTrans) {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_zgemm,
                                        &args, sizeof(struct zgemm_args)));
        }
      }
    }
    else {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i * lda];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_zgemm,
                                        &args, sizeof(struct zgemm_args)));
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_zgemm,
                                        &args, sizeof(struct zgemm_args)));
        }
      }
    }
    else {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i * lda];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_zgemm,
                                        &args, sizeof(struct zgemm_args)));
        }
      }
    }
  }

  CUresult result;
  for (size_t i = 0; i < nTasks; i++)
    CU_ERROR_CHECK(cuTaskDestroy(tasks[i], &result));

  free(tasks);

  return result;
}
