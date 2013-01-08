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

struct cgemm_plan {
  CBlasTranspose transA, transB;
  CUmodule module;
  CUstream compute, copy;
  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc, mb, nb, kb;
};

static CUresult init(const void * args) {
  struct cgemm_plan * plan = *(struct cgemm_plan **)args;

  // Load the module
  CU_ERROR_CHECK(cuModuleLoad(&plan->module, "cgemm.fatbin"));

  // Create two streams - one for compute, the other for copy
  CU_ERROR_CHECK(cuStreamCreate(&plan->compute, 0));
  CU_ERROR_CHECK(cuStreamCreate(&plan->copy, 0));

  // Allocate temporary memory for C,...
  CU_ERROR_CHECK(cuMemAllocPitch(&plan->C, &plan->ldc, plan->mb * sizeof(float complex),
                                 plan->nb, sizeof(float complex)));
  plan->ldc /= sizeof(float complex);

  // ...A...
  if (plan->transA == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A0, &plan->lda, plan->mb * sizeof(float complex),
                                   plan->kb, sizeof(float complex)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A1, &plan->lda, plan->mb * sizeof(float complex),
                                   plan->kb, sizeof(float complex)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A0, &plan->lda, plan->kb * sizeof(float complex),
                                   plan->mb, sizeof(float complex)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->A1, &plan->lda, plan->kb * sizeof(float complex),
                                   plan->mb, sizeof(float complex)));
  }
  plan->lda /= sizeof(float complex);

  // ...and B
  if (plan->transB == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B0, &plan->ldb, plan->kb * sizeof(float complex),
                                   plan->nb, sizeof(float complex)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B1, &plan->ldb, plan->kb * sizeof(float complex),
                                   plan->nb, sizeof(float complex)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B0, &plan->ldb, plan->nb * sizeof(float complex),
                                   plan->kb, sizeof(float complex)));
    CU_ERROR_CHECK(cuMemAllocPitch(&plan->B1, &plan->ldb, plan->nb * sizeof(float complex),
                                   plan->kb, sizeof(float complex)));
  }
  plan->ldb /= sizeof(float complex);

  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * args) {
  struct cgemm_plan * plan = (struct cgemm_plan *)args;

  // Free temporary memory
  CU_ERROR_CHECK(cuMemFree(plan->C));
  CU_ERROR_CHECK(cuMemFree(plan->B0));
  CU_ERROR_CHECK(cuMemFree(plan->B1));
  CU_ERROR_CHECK(cuMemFree(plan->A0));
  CU_ERROR_CHECK(cuMemFree(plan->A1));

  // Destroy the streams (this is asynchronous)
  CU_ERROR_CHECK(cuStreamDestroy(plan->copy));
  CU_ERROR_CHECK(cuStreamDestroy(plan->compute));

  // Unload the module
  CU_ERROR_CHECK(cuModuleUnload(plan->module));

  return CUDA_SUCCESS;
}

struct __cumultigpucblasconfig_st {
  CUmultiGPU mGPU;
  struct cgemm_plan * plans;
  size_t mb, nb, kb;
};

CUresult cuMultiGPUCBlasConfigCreate(CUmultiGPUCBlasConfig * config, CUmultiGPU mGPU,
                                     CBlasTranspose transA, CBlasTranspose transB,
                                     size_t mb, size_t nb, size_t kb) {
  if ((*config = malloc(sizeof(struct __cumultigpucblasconfig_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*config)->mGPU = mGPU;
  (*config)->mb = mb;
  (*config)->nb = nb;
  (*config)->kb = kb;

  int n = cuMultiGPUGetContextCount(mGPU);
  if (((*config)->plans = malloc((size_t)n * sizeof(struct cgemm_plan))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  for (int i = 0; i < n; i++) {
    struct cgemm_plan * plan = &(*config)->plans[i];
    plan->transA = transA;
    plan->transB = transB;
    plan->mb = mb;
    plan->nb = nb;
    plan->kb = kb;

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &plan, sizeof(struct cgemm_plan *)));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUCBlasConfigDestroy(CUmultiGPUCBlasConfig config) {
  int n = cuMultiGPUGetContextCount(config->mGPU);
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &config->plans[i], sizeof(struct cgemm_plan)));
    CU_ERROR_CHECK(cuMultiGPURunTask(config->mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

size_t cuMultiGPUCBlasConfigRows(CUmultiGPUCBlasConfig config) { return config->mb; }
size_t cuMultiGPUCBlasConfigColumns(CUmultiGPUCBlasConfig config) { return config->nb; }
size_t cuMultiGPUCBlasConfigInner(CUmultiGPUCBlasConfig config) { return config->kb; }

struct cgemm_args {
  struct cgemm_plan * plan;
  const float complex * A, * B;
  float complex * C;
  size_t m, n, k, lda, ldb, ldc;
  float complex alpha, beta;
  CBlasTranspose transA, transB;
};

static CUresult background_cgemm(const void * a) {
  struct cgemm_args * args = (struct cgemm_args *)a;
  struct cgemm_plan * plan = args->plan;

  // Copy C onto the device using the compute stream
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->C, plan->ldc, 0, 0,
                                     args->C, args->ldc, 0, 0,
                                     args->m, args->n, sizeof(float complex), plan->compute));

  // Perform C *= beta on the compute stream to ensure C has finished copying
  CU_ERROR_CHECK(cuCgemm(plan->module, CBlasNoTrans, CBlasNoTrans,
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
                                           args->m, lb, sizeof(float complex), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(float complex), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, 0, l + plan->kb,
                                               args->m, lb, sizeof(float complex), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, l + plan->kb, 0,
                                               lb, args->n, sizeof(float complex), plan->copy));

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
                                           lb, args->m, sizeof(float complex), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(float complex), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, l + plan->kb, 0,
                                               lb, args->m, sizeof(float complex), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, l + plan->kb, 0,
                                               lb, args->n, sizeof(float complex), plan->copy));

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
                                           args->m, lb, sizeof(float complex), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(float complex), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, 0, l + plan->kb,
                                               args->m, lb, sizeof(float complex), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, 0, l + plan->kb,
                                               args->n, lb, sizeof(float complex), plan->copy));

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
                                           lb, args->m, sizeof(float complex), plan->compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B0, plan->ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(float complex), plan->compute));

        for (size_t l = 0; l < args->k; l += plan->kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuCgemm(plan->module, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, plan->kb),
                                 args->alpha, plan->A0, plan->lda, plan->B0, plan->ldb,
                                 one, plan->C, plan->ldc, plan->compute));

          // If there is more work to do
          if (l + plan->kb < args->k) {
            const size_t lb = min(args->k - l - plan->kb, plan->kb);
            // Copy the next blocks of A and B on the opposite stream from the cgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->A1, plan->lda, 0, 0,
                                               args->A, args->lda, l + plan->kb, 0,
                                               lb, args->m, sizeof(float complex), plan->copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(plan->B1, plan->ldb, 0, 0,
                                               args->B, args->ldb, 0, l + plan->kb,
                                               args->n, lb, sizeof(float complex), plan->copy));

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
                                     args->m, args->n, sizeof(float complex), plan->compute));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUCgemm(CUmultiGPUCBlasConfig config,
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

  if (m < config->mb && n < config->nb) {
    cgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  CUtask task;
  CUtaskqueue queue;
  CU_ERROR_CHECK(cuTaskQueueCreate(&queue, ((m + config->mb - 1) / config->mb) *
                                           ((n + config->nb - 1) / config->nb)));

  int t = 0;
  int nThreads = cuMultiGPUGetContextCount(config->mGPU);

  struct cgemm_args args = { .transA = transA, .transB = transB,
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
          args.plan = &config->plans[t];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
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
          args.plan = &config->plans[t];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
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
          args.plan = &config->plans[t];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
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
          args.plan = &config->plans[t];
          CU_ERROR_CHECK(cuTaskCreate(&task, background_cgemm, &args, sizeof(struct cgemm_args)));
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
