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

struct sgemm_data {
  CUmodule module;
  CUstream compute, copy;
  CUdeviceptr A0, A1, B0, B1, C;
  size_t m, n, k, lda, ldb, ldc;
  CBlasTranspose transA, transB;
};

static CUresult init(const void * a) {
  struct sgemm_data * data = *(struct sgemm_data **)a;

  // Load the sgemm module
  CU_ERROR_CHECK(cuModuleLoad(&data->module, "sgemm.fatbin"));

  // Create separate streams for concurrent copy and execute
  CU_ERROR_CHECK(cuStreamCreate(&data->compute, 0));
  CU_ERROR_CHECK(cuStreamCreate(&data->copy, 0));

  // Allocate C (always m * n)
  CU_ERROR_CHECK(cuMemAllocPitch(&data->C, &data->ldc,
                                 data->m * sizeof(float), data->n,
                                 sizeof(float)));
  data->ldc /= sizeof(float);

  if (data->transA == CBlasNoTrans) {
    // A is m * k
    CU_ERROR_CHECK(cuMemAllocPitch(&data->A0, &data->lda,
                                   data->m * sizeof(float), data->k,
                                   sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&data->A1, &data->lda,
                                   data->m * sizeof(float), data->k,
                                   sizeof(float)));
    data->lda /= sizeof(float);
  }
  else {
    // A is k * m
    CU_ERROR_CHECK(cuMemAllocPitch(&data->A0, &data->lda,
                                   data->k * sizeof(float), data->m,
                                   sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&data->A1, &data->lda,
                                   data->k * sizeof(float), data->m,
                                   sizeof(float)));
    data->lda /= sizeof(float);
  }

  if (data->transB == CBlasNoTrans) {
    // B is k * n
    CU_ERROR_CHECK(cuMemAllocPitch(&data->B0, &data->ldb,
                                   data->k * sizeof(float), data->n,
                                   sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&data->B1, &data->ldb,
                                   data->k * sizeof(float), data->n,
                                   sizeof(float)));
    data->ldb /= sizeof(float);
  }
  else {
    // B is n * k
    CU_ERROR_CHECK(cuMemAllocPitch(&data->B0, &data->ldb,
                                   data->n * sizeof(float), data->k,
                                   sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&data->B1, &data->ldb,
                                   data->n * sizeof(float), data->k,
                                   sizeof(float)));
    data->ldb /= sizeof(float);
  }

  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * a) {
  struct sgemm_data * data = *(struct sgemm_data **)a;

  // Free A, B and C
  CU_ERROR_CHECK(cuMemFree(data->A0));
  CU_ERROR_CHECK(cuMemFree(data->A1));
  CU_ERROR_CHECK(cuMemFree(data->B0));
  CU_ERROR_CHECK(cuMemFree(data->B1));
  CU_ERROR_CHECK(cuMemFree(data->C));

  // Destroy the streams
  CU_ERROR_CHECK(cuStreamDestroy(data->compute));
  CU_ERROR_CHECK(cuStreamDestroy(data->copy));

  // Unload the module
  CU_ERROR_CHECK(cuModuleUnload(data->module));

  return CUDA_SUCCESS;
}

struct sgemm_args {
  const struct sgemm_data * data;
  size_t m, n, k, lda, ldb, ldc;
  const float * A, * B;
  float * C;
  float alpha, beta;
};

static CUresult background_sgemm(const void * a) {
  struct sgemm_args * args = (struct sgemm_args *)a;

  // Unpack the arguments
  const struct sgemm_data * data = args->data;
  CUmodule module = data->module;
  const CBlasTranspose transA = data->transA, transB = data->transB;
  CUdeviceptr A0 = data->A0, A1 = data->A1,
              B0 = data->B0, B1 = data->B1, dC = data->C;
  const size_t dlda = data->lda, dldb = data->ldb, dldc = data->ldc;
  const size_t kb = data->k;

  const size_t m = args->m, n = args->n, k = args->k;
  float alpha = args->alpha, beta = args->beta;
  const float * A = args->A, * B = args->B;
  float * C = args->C;
  const size_t lda = args->lda, ldb = args->ldb, ldc = args->ldc;

  // Create copies of the streams as they get swapped
  CUstream compute = data->compute;
  CUstream copy = data->copy;

  // Copy C onto the device using the compute stream
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(dC, dldc, 0, 0, C, ldc, 0, 0,
                                     m, n, sizeof(float), compute));

  // Perform C *= beta on the compute stream to ensure C has finished copying
  CU_ERROR_CHECK(cuSgemm(module, CBlasNoTrans, CBlasNoTrans, m, n, 0,
                         zero, 0, ldc, 0, 0, beta, dC, dldc, compute));

  // Can exit early if alpha * op(A) * op(B) will evaluate to zero
  if (alpha != zero && k > 0) {

    // Perform C += alpha * op(A) * op(B)
    if (transB == CBlasNoTrans) {
      if (transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, dlda, 0, 0, A, lda, 0, 0,
                                           m, lb, sizeof(float), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, dldb, 0, 0, B, ldb, 0, 0,
                                           lb, n, sizeof(float), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, dlda, B0, dldb, one, dC, dldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, dlda, 0, 0, A, lda, 0, l + kb,
                                               m, lb, sizeof(float), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, dldb, 0, 0, B, ldb, l + kb, 0,
                                               lb, n, sizeof(float), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, dlda, 0, 0, A, lda, 0, 0,
                                           lb, m, sizeof(float), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, dldb, 0, 0, B, ldb, 0, 0,
                                           lb, n, sizeof(float), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, dlda, B0, dldb, one, dC, dldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, dlda, 0, 0, A, lda, l + kb, 0,
                                               lb, m, sizeof(float), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, dldb, 0, 0, B, ldb, l + kb, 0,
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
      if (transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, dlda, 0, 0, A, lda, 0, 0,
                                           m, lb, sizeof(float), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, dldb, 0, 0, B, ldb, 0, 0,
                                           n, lb, sizeof(float), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, dlda, B0, dldb, one, dC, dldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, dlda, 0, 0, A, lda, 0, l + kb,
                                               m, lb, sizeof(float), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, dldb, 0, 0, B, ldb, 0, l + kb,
                                               n, lb, sizeof(float), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, dlda, 0, 0, A, lda, 0, 0,
                                           lb, m, sizeof(float), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, dldb, 0, 0, B, ldb, 0, 0,
                                           n, lb, sizeof(float), compute));

        for (size_t l = 0; l < k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(module, transA, transB, m, n, min(k - l, kb),
                                 alpha, A0, dlda, B0, dldb, one, dC, dldc, compute));

          // If there is more work to do
          if (l + kb < k) {
            const size_t lb = min(k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, dlda, 0, 0, A, lda, l + kb, 0,
                                               lb, m, sizeof(float), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, dldb, 0, 0, B, ldb, 0, l + kb,
                                               n, lb, sizeof(float), copy));

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
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(C, ldc, 0, 0, dC, dldc, 0, 0,
                                     m, n, sizeof(float), compute));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSgemm(CUthread * threads, int nThreads,
                         CBlasTranspose transA, CBlasTranspose transB,
                         size_t m, size_t n, size_t k,
                         float alpha, const float * restrict A, size_t lda,
                         const float * restrict B, size_t ldb,
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
   * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
   * to mask memory latency (64 * 3 = 192 threads/6 warps).
   * A maximum of 8 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have 30 * 8 = 240 blocks sent to the GPU.  This requires a 10x24, 12x20,
   * 15x16, etc. block size here.
   * 10x24 is chosen to retain the m >> n behaviour needed for SPOTRF('L',..).
   * mb = 10 * 64 = 640
   * nb = 24 * 16 = 384
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  kb >= 512 gives ~400GFlops/s.  This requires (640 * 384 + 2 *
   * 512 * (640 + 384)) * 4 = 5056kB of graphics memory.
   *
   * These block sizes give a bandwidth reduction of 2 / (1/640 + 1/384) = 480
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA == CBlasNoTrans is
   * (400 * 10^9) / (6 * 1024^3 / sizeof(float)) = 248.35
   *
   * When transA != CBlasNoTrans each GPU MP processes blocks of 32x32 using 64
   * threads per block.
   * There are 30 MPs on the GTX 280 and each requires a minimum of 3 blocks
   * to mask memory latency (64 * 3 = 192 threads/6 warps).
   * A maximum of 6 blocks will fit on each MP concurrently due to shared memory
   * and register requirements.  Best performance should therefore occur when we
   * have 30 * 6 = 180 blocks sent to the GPU.  This requires a 9x20, 12x15,
   * 6x30, etc. block size here.
   * 9x20 is chosen to retain the m << n behaviour needed for SPOTRF('U',..).
   * mb =  9 * 32 = 288
   * nb = 20 * 32 = 640
   * kb defines the amount of work done by each thread and the memory (and
   * bandwidth) needed for A and B so needs to be tuned to give maximum
   * performance.  288 <= kb <= 448 gives 330-345GFlops/s.  This requires (288 *
   * 640 + 2 * 288 * (288 + 640) * 4 = 1764kB of graphics memory.
   *
   * These block sizes give a bandwidth reduction of 2 / (1/288 + 1/640) = 397.24
   *
   * Bandwidth between host and device is 6 GB/s each way
   *
   * FLOP:word ratio for transA != CBlasNoTrans is
   * (330 * 10^9) / (6 * 1024^3 / sizeof(float)) = 214.20
   *
   */
  const size_t mb = (transA == CBlasNoTrans) ? 640 : 288;
  const size_t nb = (transA == CBlasNoTrans) ? 384 : 640;
  const size_t kb = (transA == CBlasNoTrans) ? 512 : 288;

  if (m < mb && n < nb) {
    sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, ((m + mb - 1) / mb) * ((n + nb - 1) / nb) + (size_t)nThreads * 2));
  int t = 0;

  struct sgemm_args args = { .k = k,
                             .alpha = alpha, .lda = lda, .ldb = ldb,
                             .beta = beta, .ldc = ldc };

  struct sgemm_data * data[nThreads];
  for (int i = 0; i < nThreads; i++) {
    if ((data[i] = malloc(sizeof(struct sgemm_data))) == NULL) {
      cuTaskQueueDestroy(queue);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    data[i]->m = mb;
    data[i]->n = nb;
    data[i]->k = kb;
    data[i]->transA = transA;
    data[i]->transB = transB;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &data[i], sizeof(struct sgemm_data *)));
    CU_ERROR_CHECK(cuThreadRunTask(threads[i], task));
  }

  if (transB == CBlasNoTrans) {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          args.data = data[t];
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
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i * lda];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          args.data = data[t];
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
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += nb) {
        args.n = min(n - j, nb);
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          args.data = data[t];
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
        for (size_t i = 0; i < m; i += mb) {
          args.m = min(m - i, mb);
          args.A = &A[i * lda];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          args.data = data[t];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuThreadRunTask(threads[t++], task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
  }

  for (int i = 0; i < nThreads; i++) {
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &data[i], sizeof(struct sgemm_data *)));
    CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
    CU_ERROR_CHECK(cuThreadRunTask(threads[i], task));
  }

  CUresult result;
  while (cuTaskQueuePop(queue, &task) == CUDA_SUCCESS)
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));

  for (int i = 0; i < nThreads; i++)
    free(data[i]);

  cuTaskQueueDestroy(queue);

  return result;
}
