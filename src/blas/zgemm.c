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
    snprintf(name, 90, "_Z6zgemmNIL14CBlasTranspose%dELj64ELj4ELj16ELj%uELj%uEEv7double2S1_PKS1_S3_S3_PS1_iiiiiii", transB, bx, by);
  }
  else {
    mb =  8;
    nb = (transB == CBlasNoTrans) ?  8 : 16;
    kb = (transB == CBlasNoTrans) ?  4 :  8;
    bx = (transB == CBlasNoTrans) ?  4 :  8;
    by =  8;
    snprintf(name, 95, "_Z6zgemmTIL14CBlasTranspose%dELS0_%dELj8ELj%uELj%uELj%uELj8EEv7double2S1_PKS1_S3_S3_PS1_iiiiiii", transA, transB, nb, kb, bx);
  }

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &alpha, &beta, &A, &B, &C, &D, &lda, &ldb, &ldc, &ldd, &m, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

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
  *
  * mb = 10 * 64 = 640
  * nb = 24 *  4 =  96
  *
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  It should be a multiple of the kb block size used to unroll
  * the GPU code which in this case is 16.  kb is increased for given mb and nb
  * until the performance increase is < 1%. This happens at kb = 64 and gives
  * ~77GFlops/s.  This requires
  * (640 * 96 + 2 * 64 * (640 + 96)) * 16 = 2432kB
  * of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/640 + 1/96) = 166.96
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for transA == CBlasNoTrans is
  * (77 * 10^9) / (6 * 1024^3 / sizeof(double complex)) = 191.23
  *
  * Since the bandwidth reduction for this block size is less than the
  * FLOP:word ratio this creates a bandwidth bound algorithm.  Increasing the
  * block sizes to 1024 * 180 sends 720 (16 * 45) blocks to the GPU, or 24 to
  * each MP, which is also a multiple of 8, the maximum that will fit.
  * This gives a final configuration of:
  * mb = 16 * 64 = 1024
  * nb = 45 *  4 =  180
  * kb (after tuning run with new mb and nb) =  64
  * memory = (1024 * 180 + 2 * 64 * (1024 + 180)) * 16 = 5288kB
  * bandwidth reduction = 2 / (1/1024 + 1/180) = 306.18
  * FLOP:word ratio = (77 * 10^9) / (6 * 1024^3 / sizeof(double complex)) = 191.23
  *
  *
  * When transA != CBlasNoTrans and transB == CBlasNoTrans each GPU MP
  * processes blocks of 8x8 using 32 threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 6 blocks
  * to mask memory latency (32 * 6 = 192 threads/6 warps).
  * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 10x24, 12x20,
  * 15x16, etc. block size here.
  * 10x24 is chosen to retain the m << n behaviour needed for ZPOTRF('U',..).
  *
  * mb = 10 *  8 =  80
  * nb = 24 *  8 = 192
  *
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  It should be a multiple of the kb block size used to unroll
  * the GPU code which in this case is 4.  kb is increased for given mb and nb
  * until the performance increase is < 1%. This happens at kb = 80 and gives
  * ~60GFlops/s.  This requires
  * (80 * 192 + 2 * 80 * (80 + 192)) * 16 = 920kB
  * of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/80 + 1/192) = 112.94
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for transA != CBlasNoTrans is
  * (60 * 10^9) / (6 * 1024^3 / sizeof(double complex)) = 149.01
  *
  * Since the bandwidth reduction for this block size is less than the
  * FLOP:word ratio this creates a bandwidth bound algorithm.  Increasing the
  * block sizes to 120 * 384 sends 720 (15 * 48) blocks to the GPU, or 24 to
  * each MP, which is also a multiple of 8, the maximum that will fit.
  * This gives a final configuration of:
  * mb = 15 *  8 =  120
  * nb = 48 *  8 =  384
  * kb (after tuning run with new mb and nb) = 32
  * memory = (120 * 384 + 2 * 32 * (120 + 384)) * 16 = 1224kB
  * bandwidth reduction = 2 / (1/120 + 1/384) = 182.86
  * FLOP:word ratio = (58 * 10^9) / (6 * 1024^3 / sizeof(float complex)) = 144.04
  *
  *
  * When transA != CBlasNoTrans and transB != CBlasNoTrans each GPU MP
  * processes blocks of 8x16 using 64 threads per block.
  * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
  * to mask memory latency (64 * 3 = 192 threads/6 warps).
  * A maximum of 4 blocks will fit on each MP concurrently due to shared memory
  * and register requirements.  Best performance should therefore occur when we
  * have 30 * 4 = 120 blocks sent to the GPU.  This requires a 6x20, 8x15, 4x30,
  * etc. block size here.
  * 10x24 is chosen to retain the m << n behaviour needed for ZPOTRF('U',..).
  *
  * mb =  8 *  8 =  64
  * nb = 15 *  8 = 120
  *
  * kb defines the amount of work done by each thread and the memory (and
  * bandwidth) needed for A and B so needs to be tuned to give maximum
  * performance.  It should be a multiple of the kb block size used to unroll
  * the GPU code which in this case is 80.  kb is increased for given mb and nb
  * until the performance increase is < 1%. This happens at kb = 80 and gives
  * ~33GFlops/s.  This requires
  * (64 * 120 + 2 * 80 * (64 + 120)) * 16 = 580kB
  * of graphics memory.
  *
  * These block sizes give a bandwidth reduction of 2 / (1/64 + 1/120) = 83.48
  *
  * Bandwidth between host and device is 6 GB/s each way
  *
  * FLOP:word ratio for transA != CBlasNoTrans is
  * (33 * 10^9) / (6 * 1024^3 / sizeof(double complex)) = 81.96
  *
  */
#define MB ((transA == CBlasNoTrans) ? 1024 : (transB == CBlasNoTrans) ? 120 :  64)
#define NB ((transA == CBlasNoTrans) ?  180 : (transB == CBlasNoTrans) ? 384 : 180)
#define KB ((transA == CBlasNoTrans) ?   64 : (transB == CBlasNoTrans) ?  32 :  80)

#include "multigpuzgemm.c"

CUresult cuMultiGPUZgemm(CUthread * threads, int nThreads,
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

  const size_t mb = MB;
  const size_t nb = NB;

  if (m < mb && n < nb) {
    zgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, ((m + mb - 1) / mb) * ((n + nb - 1) / nb)));
  int t = 0;

  struct zgemm_args args = { .transA = transA, .transB = transB,
                             .k = k,
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_zgemm, &args, sizeof(struct zgemm_args)));
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_zgemm, &args, sizeof(struct zgemm_args)));
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_zgemm, &args, sizeof(struct zgemm_args)));
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
          CU_ERROR_CHECK(cuTaskCreate(&task, background_zgemm, &args, sizeof(struct zgemm_args)));
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
