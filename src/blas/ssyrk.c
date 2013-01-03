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

static const float zero = 0.0f;
static const float one = 1.0f;

void ssyrk(CBlasUplo uplo, CBlasTranspose trans,
           size_t n, size_t k,
           float alpha, const float * restrict A, size_t lda,
           float beta, float * restrict C, size_t ldc) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

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
            register float temp = alpha * A[l * lda + j];
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
            register float temp = alpha * A[l * lda + j];
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
          register float temp = zero;
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
          register float temp = zero;
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

CUresult cuSsyrk(CUmodule module, CBlasUplo uplo, CBlasTranspose trans,
                 size_t n, size_t k,
                 float alpha, CUdeviceptr A, size_t lda,
                 float beta, CUdeviceptr C, size_t ldc, CUstream stream) {
  size_t nRowA = (trans == CBlasNoTrans) ? n : k;

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

  const unsigned int mb = (trans == CBlasNoTrans) ? 64 : 32;
  const unsigned int nb = (trans == CBlasNoTrans) ? 16 : 32;
  const unsigned int kb = (trans == CBlasNoTrans) ? 16 :  8;
  const unsigned int bx = (trans == CBlasNoTrans) ? 16 :  8;
  const unsigned int by = (trans == CBlasNoTrans) ?  4 :  8;

  char name[82];
  snprintf(name, 82,
           "_Z5ssyrkIL9CBlasUplo%dEL14CBlasTranspose%dELj%uELj%uELj%uELj%uELj%uEEvPKfPfffiiii",
           uplo, trans, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &A, &C, &alpha, &beta, &lda, &ldc, &n, &k };

//   unsigned int blocks = (unsigned int)(n + nb - 1) / nb;
//   blocks = (blocks * (blocks + 1)) / 2;

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(n + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}

/**
  * When trans == CBlasNoTrans each GPU MP processes blocks of 64x16 using 64
  * threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
  * to mask memory latency (64 * 3 = 192 threads/6 warps).
  * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 8x30, 10x24, 12x20,
  * 15x16, etc. block size here.
  * 8x30 is chosen to retain the m ~ n behaviour needed for SSYRK.
  * mb =  8 * 64 = 512
  * nb = 30 * 16 = 480
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  kb >= 192 gives ~380GFlops/s.  This requires (512 * 480 + 2 *
  * 192 * (512 + 480)) * 4 = 1682kB of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/512 + 1/480) = 495.48
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for trans == CBlasNoTrans is
  * (380 * 10^9) / (6 * 1024^3 / sizeof(float)) = 235.94
  *
  * When trans != CBlasNoTrans each GPU MP processes blocks of 32x32 using 64
  * threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
  * to mask memory latency (64 * 3 = 192 threads/6 warps).
  * A maximum of 6 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 6 = 180 blocks sent to the GPU.  This requires a 9x20, 12x15,
  * 6x30, etc. block size here.
  * 12x15 is chosen to retain the m << n behaviour needed for SPOTRF('U',..).
  * mb = 12 * 32 = 384
  * nb = 15 * 32 = 480
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  kb = 128 gives 300GFlops/s.  This requires (384 * 480 + 2 *
  * 128 * (384 + 480) * 4 = 1584kB of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/384 + 1/480) = 426.67
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for trans != CBlasNoTrans is
  * (300 * 10^9) / (6 * 1024^3 / sizeof(float)) = 186.26
  *
  */
#define MB ((transA == CBlasNoTrans) ? 512 : 384)
#define NB ((transA == CBlasNoTrans) ? 480 : 480)
#define KB ((transA == CBlasNoTrans) ? 192 : 128)

#include "multigpusgemm.c"

CUresult cuMultiGPUSsyrk(CUthread * threads, int nThreads,
                         CBlasUplo uplo, CBlasTranspose transA,
                         size_t n, size_t k,
                         float alpha, const float * restrict A, size_t lda,
                         float beta, float * restrict C, size_t ldc) {
  size_t nRowA = (transA == CBlasNoTrans) ? n : k;

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

  const size_t nb = NB;

  if (n < nb) {
    ssyrk(uplo, transA, n, k, alpha, A, lda, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, (((n + nb - 1) / nb) * ((n + nb - 1) / nb)) / 2));
  int t = 0;

  struct sgemm_args args = { .transA = transA, .transB = (transA == CBlasNoTrans) ? CBlasTrans : CBlasNoTrans,
                             .k = k,
                             .alpha = alpha, .lda = lda, .ldb = lda,
                             .beta = beta, .ldc = ldc };

  if (transA == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
      for (size_t j = nb; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < j; i += nb) {
          args.m = min(n - i, nb);
          args.A = &A[i];
          args.B = &A[j * lda];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
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
        for (size_t i = nb; i < n; i += nb) {
          args.m = min(n - i, nb);
          args.A = &A[i];
          args.B = &A[j * lda];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t j = nb; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < j; i += nb) {
          args.m = min(n - i, nb);
          args.A = &A[i * lda];
          args.B = &A[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
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
        for (size_t i = nb; i < n; i += nb) {
          args.m = min(n - i, nb);
          args.A = &A[i * lda];
          args.B = &A[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
  }

  for (size_t j = 0; j < n; j += nb)
    ssyrk(uplo, transA, min(n - j, nb), k,
          alpha, &A[(transA == CBlasNoTrans) ? j : j * lda], lda,
          beta, &C[j * ldc + j], ldc);

  CUresult result;
  while (cuTaskQueuePop(queue, &task) == CUDA_SUCCESS)
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));

  cuTaskQueueDestroy(queue);

  return result;
}
