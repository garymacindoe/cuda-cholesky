#include "blas.h"
#include "error.h"
#include "cutaskqueue.h"
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

static const float complex zero = 0.0f + 0.0f * I;
static const float complex one = 1.0f + 0.0f * I;

void cgemm(CBlasTranspose transA, CBlasTranspose transB,
           size_t m, size_t n, size_t k,
           float complex alpha, const float complex * restrict A, size_t lda, const float complex * restrict B, size_t ldb,
           float complex beta, float complex * restrict C, size_t ldc) {
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
            register float complex temp = alpha * B[j * ldb + l];
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
          register float complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conjf(A[i * lda + l]) * B[j * ldb + l];
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
          register float complex temp = zero;
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
            register float complex temp = alpha * conjf(B[l * ldb + j]);
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
          register float complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conjf(A[i * lda + l]) * conjf(B[l * ldb + j]);
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
          register float complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += A[i * lda + l] * conjf(B[l * ldb + j]);
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
            register float complex temp = alpha * B[l * ldb + j];
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
          register float complex temp = zero;
          for (size_t l = 0; l < k; l++)
            temp += conjf(A[i * lda + l]) * B[l * ldb + j];
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
          register float complex temp = zero;
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

CUresult cuCgemm2(CUmodule module, CBlasTranspose transA, CBlasTranspose transB,
                  size_t m, size_t n, size_t k,
                  float complex alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
                  float complex beta, CUdeviceptr C, size_t ldc, CUdeviceptr D, size_t ldd,
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
  const unsigned int nb = (transA == CBlasNoTrans) ?  8 : 16;
  const unsigned int kb = (transA == CBlasNoTrans) ? 16 :  8;
  const unsigned int bx = (transA == CBlasNoTrans) ? ((transB == CBlasNoTrans) ? 16 : 8) :  8;
  const unsigned int by = (transA == CBlasNoTrans) ? ((transB == CBlasNoTrans) ?  4 : 8) :  8;

  char name[95];
  snprintf(name, 95,
           "_Z5cgemmIL14CBlasTranspose%dELS0_%dELj%uELj%uELj%uELj%uELj%uEEvPK6float2S3_S3_PS1_S1_S1_iiiiiii",
           transA, transB, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &A, &B, &C, &D, &alpha, &beta, &lda, &ldb, &ldc, &ldd, &m, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

/**
  * When transA == CBlasNoTrans each GPU MP processes blocks of 64x8 using 64
  * threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
  * to mask memory latency (64 * 3 = 192 threads/6 warps).
  * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 10x24, 12x20,
  * 15x16, etc. block size here.
  * 10x24 is chosen to retain the m >> n behaviour needed for CPOTRF('L',..).
  *
  * mb = 10 * 64 = 640
  * nb = 24 *  8 = 192
  *
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  It should be a multiple of the kb block size used to unroll
  * the GPU code which in this case is 16.  kb is increased for given mb and nb
  * until the performance increase is < 1%. This happens at kb = 128 and gives
  * ~415GFlops/s.  This requires
  * (640 * 192 + 2 * 128 * (640 + 192)) * 8 = 2624kB
  * of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/640 + 1/192) = 295.38
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for transA == CBlasNoTrans is
  * (415 * 10^9) / (6 * 1024^3 / sizeof(float complex)) = 515.33
  *
  * Since the bandwidth reduction for this block size is less than the
  * FLOP:word ratio this creates a bandwidth bound algorithm.  Increasing the
  * block sizes to 1024 * 360 sends 720 (16 * 45) blocks to the GPU, or 24 to
  * each MP, which is also a multiple of 8, the maximum that will fit.
  * This gives a final configuration of:
  * mb = 16 * 64 = 1024
  * nb = 45 *  8 =  360
  * kb (after tuning run with new mb and nb) = 128
  * memory = (1024 * 360 + 2 * 128 * (1024 + 360)) * 8 = 5648kB
  * bandwidth reduction = 2 / (1/1024 + 1/360) = 532.72
  * FLOP:word ratio = (425 * 10^9) / (6 * 1024^3 / sizeof(float complex)) = 527.75
  *
  *
  * When transA != CBlasNoTrans each GPU MP processes blocks of 32x16 using 64
  * threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
  * to mask memory latency (64 * 3 = 192 threads/6 warps).
  * A maximum of 4 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 4 = 120 blocks sent to the GPU.  This requires a 6x20, 8x15, 4x30,
  * etc. block size here.
  * 8x15 is chosen to retain the m << n behaviour needed for CPOTRF('U',..).
  *
  * mb =  4 * 32 = 128
  * nb = 30 * 16 = 480
  *
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  It should be a multiple of the kb block size used to unroll
  * the GPU code which in this case is 8.  kb is increased for given mb and nb
  * until the performance increase is < 1%. This happens at kb = 264 and gives
  * ~305GFlops/s.  This requires
  * (128 * 480 + 2 * 264 * (128 + 480)) * 8 = 2988kB
  * of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/128 + 1/480) = 202.11
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for transA != CBlasNoTrans is
  * (305 * 10^9) / (6 * 1024^3 / sizeof(float complex)) = 378.74
  *
  * Since the bandwidth reduction for this block size is less than the
  * FLOP:word ratio this creates a bandwidth bound algorithm.  Increasing the
  * block sizes to 256 * 720 sends 360 (8 * 45) blocks to the GPU, or 12 to
  * each MP, which is also a multiple of 4, the maximum that will fit.
  * This gives a final configuration of:
  * mb =  8 * 32 =  256
  * nb = 45 * 16 =  720
  * kb (after tuning run with new mb and nb) = 128
  * memory = (256 * 720 + 2 * 128 * (256 + 720)) * 8 = 3392kB
  * bandwidth reduction = 2 / (1/1024 + 1/360) = 377.70
  * FLOP:word ratio = (300 * 10^9) / (6 * 1024^3 / sizeof(float complex)) = 372.53
  *
  */
#define MB ((transA == CBlasNoTrans) ? 1024 : 256)
#define NB ((transA == CBlasNoTrans) ?  360 : 720)
#define KB ((transA == CBlasNoTrans) ?  128 : 128)

#include "multigpucgemm.c"

CUresult cuMultiGPUCgemm(CUthread * threads, int nThreads,
                         CBlasTranspose transA, CBlasTranspose transB,
                         size_t m, size_t n, size_t k,
                         float complex alpha, const float complex * restrict A, size_t lda,
                         const float complex * restrict B, size_t ldb,
                         float complex beta, float complex * restrict C, size_t ldc) {
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

  const size_t mb = MB;
  const size_t nb = NB;

  if (m < mb && n < nb) {
    cgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, ((m + mb - 1) / mb) * ((n + nb - 1) / nb)));
  int t = 0;

  struct cgemm_args args = { .k = k,
                             .alpha = alpha, .lda = lda, .ldb = ldb,
                             .beta = beta, .ldc = ldc };

  if (transB == CBlasNoTrans) {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
  }

  CUresult result;
  while (cuTaskQueuePop(queue, &task) == CUDA_SUCCESS)
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));

  cuTaskQueueDestroy(queue);

  return result;
}
