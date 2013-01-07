#include "blas.h"
#include "error.h"
#include "../multigpu.h"
#include "../taskqueue.h"
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

struct sgemm_plan {
  CBlasTranspose transA, transB;
  CUmodule module;
  CUstream compute, copy;
  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc, mb, nb, kb;
};

static CUresult init(const void * args) {
  struct sgemm_plan * plan = (struct sgemm_plan *)args;

  // Load the module
  CU_ERROR_CHECK(cuModuleLoad(&plan->module, "sgemm.fatbin"));

  // Create two streams - one for compute, the other for copy
  CU_ERROR_CHECK(cuStreamCreate(&plan->compute, 0));
  CU_ERROR_CHECK(cuStreamCreate(&plan->copy, 0));

  // Allocate temporary memory for C,...
  CU_ERROR_CHECK(cuMemAllocPitch(&plan->C, &plan->ldc, plan->mb * sizeof(float),
                                 plan->nb, sizeof(float)));
  plan->ldc /= sizeof(float);

  // ...A...
  if (plan->transA == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A0, &plan->lda, plan->mb * sizeof(float),
                                   plan->kb, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A1, &plan->lda, plan->mb * sizeof(float),
                                   plan->kb, sizeof(float)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A0, &plan->lda, plan->kb * sizeof(float),
                                   plan->mb, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A1, &plan->lda, plan->kb * sizeof(float),
                                   plan->mb, sizeof(float)));
  }
  plan->lda /= sizeof(float);

  // ...and B
  if (plan->transB == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B0, &plan->ldb, plan->kb * sizeof(float),
                                   plan->nb, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B1, &plan->ldb, plan->kb * sizeof(float),
                                   plan->nb, sizeof(float)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B0, &plan->ldb, plan->nb * sizeof(float),
                                   plan->kb, sizeof(float)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B1, &plan->ldb, plan->nb * sizeof(float),
                                   plan->kb, sizeof(float)));
  }
  plan->ldb /= sizeof(float);

  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * args) {
  struct sgemm_plan * plan = (struct sgemm_plan *)args;

  // Free temporary memory
  CU_ERROR_CHECK(cuMemFree(plan->C));
  CU_ERROR_CHECK(cuMemFree(plan->B0));
  CU_ERROR_CHECK(cuMemFree(plan->B1));
  CU_ERROR_CHECK(cuMemFree(plan->A0));
  CU_ERROR_CHECK(cuMemFree(plan->A1));

  // Destroy the streams (this is asynchronous)
  CU_ERROR_CHECK(cuStreamDestroy(plan->compute));
  CU_ERROR_CHECK(cuStreamDestroy(plan->copy));

  // Unload the module
  CU_ERROR_CHECK(cuModuleUnload(plan->module));

  return CUDA_SUCCESS;
}

struct __cumultigpusconfig_st {
  CUmultiGPU mGPU;
  struct sgemm_plan * plans;
  size_t mb, nb;
};

CUresult cuMultiGPUSConfigCreate(CUmultiGPUSConfig * config, CUmultiGPU mGPU,
                                 CBlasTranspose transA, CBlasTranspose transB,
                                 size_t mb, size_t nb, size_t kb) {
  if ((*config = malloc(sizeof(struct __cumultigpusconfig_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*config)->mGPU = mGPU;
  (*config)->mb = mb;
  (*config)->nb = nb;

  int n = cuMultiGPUGetContextCount(mGPU);
  if (((*config)->plans = malloc((size_t)n * sizeof(struct sgemm_plan))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  for (int i = 0; i < n; i++) {
    (*config)->plans[i].transA = transA;
    (*config)->plans[i].transB = transB;
    (*config)->plans[i].mb = mb;
    (*config)->plans[i].nb = nb;
    (*config)->plans[i].kb = kb;

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &(*config)->plans[i], sizeof(struct sgemm_plan)));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSConfigDestroy(CUmultiGPUSConfig config) {
  int n = cuMultiGPUGetContextCount(config->mGPU);
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &config->plans[i], sizeof(struct sgemm_plan)));
    CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

struct sgemm_args {
  struct sgemm_plan * plan;
  const float * A, * B;
  float * C;
  size_t m, n, k, lda, ldb, ldc;
  float alpha, beta;
  CBlasTranspose transA, transB;
};

static CUresult background_sgemm(const void * a) {
  struct sgemm_args * args = (struct sgemm_args *)a;
  struct sgemm_plan * plan = args->plan;

  // Copy C onto the device using the compute stream
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->C, plan->ldc, 0, 0,
                                     args->C, args->ldc, 0, 0,
                                     args->m, args->n, sizeof(float), plan->compute));

  // Perform C *= beta on the compute stream to ensure C has finished copying
  CU_ERROR_CHECK(cuSgemm(plan->module, CBlasNoTrans, CBlasNoTrans,
                         args->m, args->n, 0,
                         zero, 0, plan->ldc, 0, 0,
                         args->beta, plan->C, plan->ldc, plan->compute));

  // Can exit early if alpha * op(A) * op(B) will evaluate to zero
  if (args->alpha != zero && args->k > 0) {

    // Perform C += alpha * op(A) * op(B)
    if (args->transB == CBlasNoTrans) {
      if (args->transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, plan->kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A0, plan->lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           args->m, lb, sizeof(float), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(float), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, 0, l + plan->kb,
                                               args->m, lb, sizeof(float), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, l + plan->kb, 0,
                                               lb, args->n, sizeof(float), plan->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = plan->compute; plan->compute = plan->copy; plan->copy = stream;
            CUdeviceptr ptr = plan->A0; plan->A0 = plan->A1; plan->A1 = ptr;
            ptr = plan->B0; plan->B0 = plan->B1; plan->B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, plan->kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A0, plan->lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           lb, args->m, sizeof(float), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(float), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, l + plan->kb, 0,
                                               lb, args->m, sizeof(float), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, l + plan->kb, 0,
                                               lb, args->n, sizeof(float), plan->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = plan->compute; plan->compute = plan->copy; plan->copy = stream;
            CUdeviceptr ptr = plan->A0; plan->A0 = plan->A1; plan->A1 = ptr;
            ptr = plan->B0; plan->B0 = plan->B1; plan->B1 = ptr;
          }
        }
      }
    }
    else {
      if (args->transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, plan->kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A0, plan->lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           args->m, lb, sizeof(float), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(float), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, 0, l + plan->kb,
                                               args->m, lb, sizeof(float), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, 0, l + plan->kb,
                                               args->n, lb, sizeof(float), plan->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = plan->compute; plan->compute = plan->copy; plan->copy = stream;
            CUdeviceptr ptr = plan->A0; plan->A0 = plan->A1; plan->A1 = ptr;
            ptr = plan->B0; plan->B0 = plan->B1; plan->B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, plan->kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A0, plan->lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           lb, args->m, sizeof(float), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(float), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuSgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the sgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, l + plan->kb, 0,
                                               lb, args->m, sizeof(float), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, 0, l + plan->kb,
                                               args->n, lb, sizeof(float), plan->copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = plan->compute; plan->compute = plan->copy; plan->copy = stream;
            CUdeviceptr ptr = plan->A0; plan->A0 = plan->A1; plan->A1 = ptr;
            ptr = plan->B0; plan->B0 = plan->B1; plan->B1 = ptr;
          }
        }
      }
    }
  }

  // Copy C back onto the host on the compute stream
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(args->C, args->ldc, 0, 0, plan->C, plan->ldc, 0, 0,
                                     args->m, args->n, sizeof(float), plan->compute));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUSgemm(CUmultiGPUSConfig config,
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
//   const size_t mb = (transA == CBlasNoTrans) ? 640 : 288;
//   const size_t nb = (transA == CBlasNoTrans) ? 384 : 640;
//   const size_t kb = (transA == CBlasNoTrans) ? 512 : 288;

  if (m < config->mb && n < config->nb) {
    sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, ((m + config->mb - 1) / config->mb) *
                                           ((n + config->nb - 1) / config->nb)));

  int t = 0;
  int nThreads = cuMultiGPUGetContextCount(config->mGPU);

  struct sgemm_args args = { .transA = transA, .transB = transB,
                             .k = k,
                             .alpha = alpha, .lda = lda, .ldb = ldb,
                             .beta = beta, .ldc = ldc };

  if (transB == CBlasNoTrans) {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += config->nb) {
        args.n = min(n - j, config->nb);
        for (size_t i = 0; i < m; i += config->mb) {
          args.m = min(m - i, config->mb);
          args.A = &A[i];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, t++, task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
    else {
      for (size_t j = 0; j < n; j += config->nb) {
        args.n = min(n - j, config->nb);
        for (size_t i = 0; i < m; i += config->mb) {
          args.m = min(m - i, config->mb);
          args.A = &A[i * lda];
          args.B = &B[j * ldb];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, t++, task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      for (size_t j = 0; j < n; j += config->nb) {
        args.n = min(n - j, config->nb);
        for (size_t i = 0; i < m; i += config->mb) {
          args.m = min(m - i, config->mb);
          args.A = &A[i];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, t++, task));
          if (t == nThreads)
            t = 0;
        }
      }
    }
    else {
      for (size_t j = 0; j < n; j += config->nb) {
        args.n = min(n - j, config->nb);
        for (size_t i = 0; i < m; i += config->mb) {
          args.m = min(m - i, config->mb);
          args.A = &A[i * lda];
          args.B = &B[j];
          args.C = &C[j * ldc + i];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_sgemm, &args, sizeof(struct sgemm_args)));
          CU_ERROR_CHECK(cuTaskQueuePush(queue, task));
          CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, t++, task));
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
