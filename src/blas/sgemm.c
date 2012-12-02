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

static const float zero = 0.0f;
static const float one = 1.0f;

void sgemm(CBlasTranspose transA, CBlasTranspose transB,
           size_t m, size_t n, size_t k,
           float alpha, const float * restrict A, size_t lda, const float * restrict B, size_t ldb,
           float beta, float * restrict C, size_t ldc) {
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
            register float temp = alpha * B[j * ldb + l];
            for (size_t i = 0; i < m; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
          register float temp = zero;
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
            register float temp = alpha * B[l * ldb + j];
            for (size_t i = 0; i < m; i++)
              C[j * ldc + i] += temp * A[l * lda + i];
          }
        }
      }
    }
    else {
#pragma omp parallel for
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++) {
          register float temp = zero;
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

CUresult cuSgemm2(CUmodule module, CBlasTranspose transA, CBlasTranspose transB,
                  size_t m, size_t n, size_t k,
                  float alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
                  float beta, CUdeviceptr C, size_t ldc, CUdeviceptr D, size_t ldd,
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

  const unsigned int mb = (transA == CBlasNoTrans) ? 64 : 32;
  const unsigned int nb = (transA == CBlasNoTrans) ? 16 : 32;
  const unsigned int kb = (transA == CBlasNoTrans) ? 16 :  8;
  const unsigned int bx = (transA == CBlasNoTrans) ? 16 :  8;
  const unsigned int by = (transA == CBlasNoTrans) ?  4 :  8;

  char name[84];
  snprintf(name, 84,
           "_Z5sgemmIL14CBlasTranspose%dELS0_%dELj%uELj%uELj%uELj%uELj%uEEvPKfS2_S2_Pfffiiiiiii",
           transA, transB, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &A, &B, &C, &D, &alpha, &beta, &lda, &ldb, &ldc, &ldd, &m, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

struct sgemm_args {
  CBlasTranspose transA, transB;
  size_t m, n, k;
  float alpha; const float * A; size_t lda; const float * B; size_t ldb;
  float beta; float * C; size_t ldc;
};

static CUresult background_sgemm(const void * a) {
  struct sgemm_args * args = (struct sgemm_args *)a;

  const CBlasTranspose transA = args->transA;
  const CBlasTranspose transB = args->transB;
  const size_t m = args->m;
  const size_t n = args->n;
  const size_t k = args->k;
  const float alpha = args->alpha;
  const float beta = args->beta;

  if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc;

  const size_t kb = (transA == CBlasNoTrans) ? 512 : 320;

  // Load the sgemm module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoad(&module, "sgemm.fatbin"));

  // Create separate streams for concurrent copy and execute
  CUstream compute, copy;
  CU_ERROR_CHECK(cuStreamCreate(&compute, 0));
  CU_ERROR_CHECK(cuStreamCreate(&copy, 0));

  // Allocate C
  CU_ERROR_CHECK(cuMemAllocPitch(&C, &ldc, m * sizeof(float), n, sizeof(float)));
  ldc /= sizeof(float);

  // Copy C onto the device
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, 0, 0,
                                     args->C, args->ldc, 0, 0,
                                     m, n, sizeof(float), compute));

  if (transB == CBlasNoTrans) {
    // B is k * n
    CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, kb * sizeof(float), n, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, kb * sizeof(float), n, sizeof(float)));
    ldb /= sizeof(float);

    if (transA == CBlasNoTrans) {
      // A is m * k
      CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, m * sizeof(float), kb, sizeof(float)));
      CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, m * sizeof(float), kb, sizeof(float)));
      lda /= sizeof(float);

      // Copy A and B onto the device asynchronously on the same stream as C
      const size_t lb = min(k, kb);
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                         args->A, args->lda, 0, 0,
                                         m, lb, sizeof(float), compute));
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                         args->B, args->ldb, 0, 0,
                                         lb, n, sizeof(float), compute));

      for (size_t l = 0; l < k; l += kb) {
        // Compute C on the same stream as the copies to ensure they have finished first
        CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                               alpha, A0, lda, B0, ldb, beta, C, ldc, compute));

        CU_ERROR_CHECK(cuStreamSynchronize(compute));

        // If there is more work to do
        if (l + kb < k) {
          const size_t lb = min(k - l - kb, kb);
          // Copy the next blocks of A and B on the opposite stream from the sgemm
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                             args->A, args->lda, 0, l + kb,
                                             m, lb, sizeof(float), copy));
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                             args->B, args->ldb, l + kb, 0,
                                             lb, n, sizeof(float), copy));

          CU_ERROR_CHECK(cuStreamSynchronize(copy));

          // Swap the streams and pointers so that the compute starts after the copy
          CUstream stream = compute; compute = copy; copy = stream;
          CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
          ptr = B0; B0 = B1; B1 = ptr;
        }
      }
    }
    else {
      // A is k * m
      CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, kb * sizeof(float), m, sizeof(float)));
      CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, kb * sizeof(float), m, sizeof(float)));
      lda /= sizeof(float);

      // Copy A and B onto the device asynchronously on the same stream as C
      const size_t lb = min(k, kb);
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                         args->A, args->lda, 0, 0,
                                         lb, m, sizeof(float), compute));
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                         args->B, args->ldb, 0, 0,
                                         lb, n, sizeof(float), compute));

      for (size_t l = 0; l < k; l += kb) {
        // Compute C on the same stream as the copies to ensure they have finished first
        CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                               alpha, A0, lda, B0, ldb, beta, C, ldc, compute));

        // If there is more work to do
        if (l + kb < k) {
          const size_t lb = min(k - l - kb, kb);
          // Copy the next blocks of A and B on the opposite stream from the sgemm
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                             args->A, args->lda, l + kb, 0,
                                             lb, m, sizeof(float), copy));
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                             args->B, args->ldb, l + kb, 0,
                                             lb, n, sizeof(float), copy));

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
    CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, n * sizeof(float), kb, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, n * sizeof(float), kb, sizeof(float)));
    ldb /= sizeof(float);

    if (transA == CBlasNoTrans) {
      // A is m * k
      CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, m * sizeof(float), kb, sizeof(float)));
      CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, m * sizeof(float), kb, sizeof(float)));
      lda /= sizeof(float);

      // Copy A and B onto the device asynchronously on the same stream as C
      const size_t lb = min(k, kb);
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                         args->A, args->lda, 0, 0,
                                         m, lb, sizeof(float), compute));
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                         args->B, args->ldb, 0, 0,
                                         n, lb, sizeof(float), compute));

      for (size_t l = 0; l < k; l += kb) {
        // Compute C on the same stream as the copies to ensure they have finished first
        CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                               alpha, A0, lda, B0, ldb, beta, C, ldc, compute));

        // If there is more work to do
        if (l + kb < k) {
          const size_t lb = min(k - l - kb, kb);
          // Copy the next blocks of A and B on the opposite stream from the sgemm
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                             args->A, args->lda, 0, l + kb,
                                             m, lb, sizeof(float), copy));
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                             args->B, args->ldb, 0, l + kb,
                                             n, lb, sizeof(float), copy));

          // Swap the streams and pointers so that the compute starts after the copy
          CUstream stream = compute; compute = copy; copy = stream;
          CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
          ptr = B0; B0 = B1; B1 = ptr;
        }
      }
    }
    else {
      // A is k * m
      CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, kb * sizeof(float), m, sizeof(float)));
      CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, kb * sizeof(float), m, sizeof(float)));
      lda /= sizeof(float);

      // Copy A and B onto the device asynchronously on the same stream as C
      const size_t lb = min(k, kb);
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                         args->A, args->lda, 0, 0,
                                         lb, m, sizeof(float), compute));
      CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                         args->B, args->ldb, 0, 0,
                                         n, lb, sizeof(float), compute));

      for (size_t l = 0; l < k; l += kb) {
        // Compute C on the same stream as the copies to ensure they have finished first
        CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                               alpha, A0, lda, B0, ldb, beta, C, ldc, compute));

        // If there is more work to do
        if (l + kb < k) {
          const size_t lb = min(k - l - kb, kb);
          // Copy the next blocks of A and B on the opposite stream from the sgemm
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                             args->A, args->lda, l + kb, 0,
                                             lb, m, sizeof(float), copy));
          CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                             args->B, args->ldb, 0, l + kb,
                                             n, lb, sizeof(float), copy));

          // Swap the streams and pointers so that the compute starts after the copy
          CUstream stream = compute; compute = copy; copy = stream;
          CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
          ptr = B0; B0 = B1; B1 = ptr;
        }
      }
    }
  }

  // Copy C back onto the host on the compute stream
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(args->C, args->ldc, 0, 0,
                                     C, ldc, 0, 0,
                                     m, n, sizeof(float), compute));

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

CUresult cuMultiGPUSgemm(CUmultiGPU multiGPU,
                         CBlasTranspose transA, CBlasTranspose transB,
                         size_t m, size_t n, size_t k,
                         float alpha, const float * restrict A, size_t lda, const float * restrict B, size_t ldb,
                         float beta, float * restrict C, size_t ldc) {
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
   * When transA == CBlasNoTrans each GPU MP processes blocks of 64x16 using 64
   * threads per block.
   * There are 30 MPs on the GTX 280 so each requires a minimum of 3 blocks
   * to mask memory latency (64 * 3 = 192 threads/6 warps).
   * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have over 30 * 6 = 180 blocks sent to the GPU.  This requires a 9x20,
   * 12x15, 6x30, etc. block size here.
   * 9x20 is chosen to retain the m >> n behaviour needed for SPOTRF('L',..).
   * mb =  9 * 64 = 576
   * nb = 20 * 16 = 320
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  kb >= 512 gives 400GFlops/s.  This requires (576 * 320 + 2 *
   * (576 * 512 + 320 * 512)) * 4 = 4304kB of graphics memory
   *
   * These block sizes give a bandwidth reduction of 2 / (1/576 + 1/320) = 411.43
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA == CBlasNoTrans is
   * (4 * 10^9) / (6 * 1,073,741,824 / sizeof(float)) = 248.35
   *
   * When transA != CBlasNoTrans each GPU MP processes blocks of 32x32 using 64
   * threads per block.
   * There are 30 MPs on the GTX 280 so each requires a minimum of 3 blocks
   * to mask memory latency (64 * 3 = 192 threads/6 warps).
   * A maximum of 6 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have over 30 * 3 = 90 blocks sent to the GPU.  This requires a 9x10,
   * 6x15, 3x30, etc. block size here.
   * 6x15 is chosen to retain the m << n behaviour needed for SPOTRF('U',..).
   * mb =  6 * 32 = 192
   * nb = 15 * 32 = 480
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  kb >= 320 gives 250GFlops/s.  This requires (192 * 480 + 2 *
   * (129 * 320 + 320 * 480)) * 4 = 1200kB of graphics memory
   *
   * These block sizes give a bandwidth reduction of 2 / (1/192 + 1/480) = 274.29
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA != CBlasNoTrans is
   * (2.5 * 10^9) / (6 * 1,073,741,824 / sizeof(float)) = 1.55
   *
   */
  const size_t mb = (transA == CBlasNoTrans) ? 576 : 320;
  const size_t nb = (transA == CBlasNoTrans) ? 192 : 480;

  if (m < mb && n < nb) {
    sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  const size_t nTasks = ((m + mb - 1) / mb) * ((n + nb - 1) / nb);
  CUtask * tasks;
  if ((tasks = malloc(nTasks * sizeof(CUtask))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  size_t t = 0;

  struct sgemm_args args = { transA, transB,
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
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_sgemm,
                                        &args, sizeof(struct sgemm_args)));
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
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_sgemm,
                                        &args, sizeof(struct sgemm_args)));
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
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_sgemm,
                                        &args, sizeof(struct sgemm_args)));
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
          CU_ERROR_CHECK(cuTaskSchedule(&tasks[t++], multiGPU, background_sgemm,
                                        &args, sizeof(struct sgemm_args)));
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
