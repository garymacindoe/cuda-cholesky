#include "blas.h"
#include "error.h"
#include <stdio.h>
#include "handle.h"
#include "config.h"
#include "dgemm.fatbin.c"

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

void dgemm(CBlasTranspose transA, CBlasTranspose transB,
           size_t m, size_t n, size_t k,
           double alpha, const double * restrict A, size_t lda, const double * restrict B, size_t ldb,
           double beta, double * restrict C, size_t ldc) {
  const size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  const size_t nRowB = (transB == CBlasNoTrans) ? k : n;

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
            register double temp = alpha * B[j * ldb + l];
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
          register double temp = zero;
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
            register double temp = alpha * B[l * ldb + j];
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
          register double temp = zero;
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

CUresult cuDgemm2(CUBLAShandle handle, CBlasTranspose transA, CBlasTranspose transB,
                  size_t m, size_t n, size_t k,
                  double alpha, CUdeviceptr A, size_t lda, CUdeviceptr B, size_t ldb,
                  double beta, CUdeviceptr C, size_t ldc, CUdeviceptr D, size_t ldd,
                  CUstream stream) {
  const size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  const size_t nRowB = (transB == CBlasNoTrans) ? k : n;

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

  if (m == 0 || n == 0 || (C == D && (alpha == zero || k == 0) && beta == one))
    return CUDA_SUCCESS;

  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->dgemm == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->dgemm, imageBytes));

  const unsigned int mb = (transA == CBlasNoTrans) ? 64 : 32;
  const unsigned int nb = (transA == CBlasNoTrans) ?  8 : 16;
  const unsigned int kb = (transA == CBlasNoTrans) ? 16 :  8;
  const unsigned int bx = (transA == CBlasNoTrans) ? ((transB == CBlasNoTrans) ? 16 : 8) :  8;
  const unsigned int by = (transA == CBlasNoTrans) ? ((transB == CBlasNoTrans) ?  4 : 8) :  8;

  char name[83];
  snprintf(name, 83,
           "_Z5dgemmIL14CBlasTranspose%dELS0_%dELj%uELj%uELj%uELj%uELj%uEEvPKdS2_S2_Pdddiiiiiii",
           transA, transB, mb, nb, kb, bx, by);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->dgemm, name));

  void * params[] = { &A, &B, &C, &D, &alpha, &beta, &lda, &ldb, &ldc, &ldd, &m, &n, &k };

  CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)(m + mb - 1) / mb, (unsigned int)(n + nb - 1) / nb, 1,
                                bx, by, 1, 0, stream, params, NULL));

  CU_ERROR_CHECK(cuCtxPopCurrent(&handle->context));

  return CUDA_SUCCESS;
}

struct dgemm_args {
  CUBLAShandle handle;
  const double * A, * B;
  double * C;
  size_t m, n, k, lda, ldb, ldc;
  double alpha, beta;
  CBlasTranspose transA, transB;
};

static CUresult background_dgemm(const void * a) {
  struct dgemm_args * args = (struct dgemm_args *)a;
  CUBLAShandle handle = args->handle;

  // Block sizes
  const size_t mb = (args->transA == CBlasNoTrans) ? DGEMM_N_MB : DGEMM_T_MB;
  const size_t nb = (args->transA == CBlasNoTrans) ? DGEMM_N_NB : DGEMM_T_NB;
  const size_t kb = (args->transA == CBlasNoTrans) ? DGEMM_N_KB : DGEMM_T_KB;

  // Temporary device memory and streams
  CUdeviceptr A0, A1, B0, B1, C;
  size_t lda, ldb, ldc;
  CUstream copy, compute;

  // Allocate two matrices for blocks of A and B on the device and one for a
  // block of C
  if (args->transA == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, mb * sizeof(double), kb, sizeof(double)));
    CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, mb * sizeof(double), kb, sizeof(double)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&A0, &lda, kb * sizeof(double), mb, sizeof(double)));
    CU_ERROR_CHECK(cuMemAllocPitch(&A1, &lda, kb * sizeof(double), mb, sizeof(double)));
  }
  lda /= sizeof(double);

  if (args->transB == CBlasNoTrans) {
    CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, kb * sizeof(double), nb, sizeof(double)));
    CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, kb * sizeof(double), nb, sizeof(double)));
  }
  else {
    CU_ERROR_CHECK(cuMemAllocPitch(&B0, &ldb, nb * sizeof(double), kb, sizeof(double)));
    CU_ERROR_CHECK(cuMemAllocPitch(&B1, &ldb, nb * sizeof(double), kb, sizeof(double)));
  }
  ldb /= sizeof(double);

  CU_ERROR_CHECK(cuMemAllocPitch(&C, &ldc, mb * sizeof(double), nb, sizeof(double)));
  ldc /= sizeof(double);

  // Create streams
  CU_ERROR_CHECK(cuStreamCreate(&copy, CU_STREAM_NON_BLOCKING));
  CU_ERROR_CHECK(cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));

  // Copy C onto the device using the compute stream
  CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(C, ldc, 0, 0,
                                     args->C, args->ldc, 0, 0,
                                     args->m, args->n, sizeof(double), compute));

  // Perform C *= beta on the compute stream to ensure C has finished copying
  CU_ERROR_CHECK(cuDgemm(handle, CBlasNoTrans, CBlasNoTrans,
                         args->m, args->n, 0,
                         zero, 0, ldc, 0, 0,
                         args->beta, C, ldc, compute));

  // Can exit early if alpha * op(A) * op(B) will evaluate to zero
  if (args->alpha != zero && args->k > 0) {

    // Perform C += alpha * op(A) * op(B)
    if (args->transB == CBlasNoTrans) {
      if (args->transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           args->m, lb, sizeof(double), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(double), compute));

        for (size_t l = 0; l < args->k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuDgemm(handle, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, kb),
                                 args->alpha, A0, lda, B0, ldb,
                                 one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < args->k) {
            const size_t lb = min(args->k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the dgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, 0, l + kb,
                                               args->m, lb, sizeof(double), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, l + kb, 0,
                                               lb, args->n, sizeof(double), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           lb, args->m, sizeof(double), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           lb, args->n, sizeof(double), compute));

        for (size_t l = 0; l < args->k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuDgemm(handle, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, kb),
                                 args->alpha, A0, lda, B0, ldb,
                                 one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < args->k) {
            const size_t lb = min(args->k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the dgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, l + kb, 0,
                                               lb, args->m, sizeof(double), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, l + kb, 0,
                                               lb, args->n, sizeof(double), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
    }
    else {
      if (args->transA == CBlasNoTrans) {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           args->m, lb, sizeof(double), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(double), compute));

        for (size_t l = 0; l < args->k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuDgemm(handle, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, kb),
                                 args->alpha, A0, lda, B0, ldb,
                                 one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < args->k) {
            const size_t lb = min(args->k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the dgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, 0, l + kb,
                                               args->m, lb, sizeof(double), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, 0, l + kb,
                                               args->n, lb, sizeof(double), copy));

            // Swap the streams and pointers so that the compute starts after the copy
            CUstream stream = compute; compute = copy; copy = stream;
            CUdeviceptr ptr = A0; A0 = A1; A1 = ptr;
            ptr = B0; B0 = B1; B1 = ptr;
          }
        }
      }
      else {
        // Copy A and B onto the device asynchronously on the same stream as C
        const size_t lb = min(args->k, kb);
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A0, lda, 0, 0,
                                           args->A, args->lda, 0, 0,
                                           lb, args->m, sizeof(double), compute));
        CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B0, ldb, 0, 0,
                                           args->B, args->ldb, 0, 0,
                                           args->n, lb, sizeof(double), compute));

        for (size_t l = 0; l < args->k; l += kb) {
          // Compute C on the same stream as the copies to ensure they have finished first
          CU_ERROR_CHECK(cuDgemm(handle, args->transA, args->transB,
                                 args->m, args->n, min(args->k - l, kb),
                                 args->alpha, A0, lda, B0, ldb,
                                 one, C, ldc, compute));

          // If there is more work to do
          if (l + kb < args->k) {
            const size_t lb = min(args->k - l - kb, kb);
            // Copy the next blocks of A and B on the opposite stream from the dgemm
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(A1, lda, 0, 0,
                                               args->A, args->lda, l + kb, 0,
                                               lb, args->m, sizeof(double), copy));
            CU_ERROR_CHECK(cuMemcpyHtoD2DAsync(B1, ldb, 0, 0,
                                               args->B, args->ldb, 0, l + kb,
                                               args->n, lb, sizeof(double), copy));

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
  CU_ERROR_CHECK(cuMemcpyDtoH2DAsync(args->C, args->ldc, 0, 0, C, ldc, 0, 0,
                                     args->m, args->n, sizeof(double), compute));

  // Clean up temporary memory and streams
  CU_ERROR_CHECK(cuMemFree(A0));
  CU_ERROR_CHECK(cuMemFree(A1));
  CU_ERROR_CHECK(cuMemFree(B0));
  CU_ERROR_CHECK(cuMemFree(B1));
  CU_ERROR_CHECK(cuMemFree(C));

  CU_ERROR_CHECK(cuStreamDestroy(copy));
  CU_ERROR_CHECK(cuStreamDestroy(compute));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUDgemm(CUmultiGPUBLAShandle handle,
                         CBlasTranspose transA, CBlasTranspose transB,
                         size_t m, size_t n, size_t k,
                         double alpha, const double * restrict A, size_t lda,
                         const double * restrict B, size_t ldb,
                         double beta, double * restrict C, size_t ldc) {
  const size_t nRowA = (transA == CBlasNoTrans) ? m : k;
  const size_t nRowB = (transB == CBlasNoTrans) ? k : n;

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
  const size_t mb = (transA == CBlasNoTrans) ? DGEMM_N_MB : DGEMM_T_MB;
  const size_t nb = (transA == CBlasNoTrans) ? DGEMM_N_NB : DGEMM_T_NB;

  if (m < mb && n < nb) {
    dgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return CUDA_SUCCESS;
  }

  int task = 0, nTasks = (int)(((m + mb - 1) / mb) * ((n + nb - 1) / nb));
  CUtask tasks[nTasks];

  int ctx = 0;
  int nCtxs = cuMultiGPUGetContextCount(handle->mGPU);

  struct dgemm_args args = { .transA = transA, .transB = transB,
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
          args.handle = &handle->handles[ctx];
          CU_ERROR_CHECK(cuTaskCreate(&tasks[task], background_dgemm, &args, sizeof(struct dgemm_args)));
          CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, ctx++, tasks[task++]));
          if (ctx == nCtxs)
            ctx = 0;
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
          args.handle = &handle->handles[ctx];
          CU_ERROR_CHECK(cuTaskCreate(&tasks[task], background_dgemm, &args, sizeof(struct dgemm_args)));
          CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, ctx++, tasks[task++]));
          if (ctx == nCtxs)
            ctx = 0;
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
          args.handle = &handle->handles[ctx];
          CU_ERROR_CHECK(cuTaskCreate(&tasks[task], background_dgemm, &args, sizeof(struct dgemm_args)));
          CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, ctx++, tasks[task++]));
          if (ctx == nCtxs)
            ctx = 0;
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
          args.handle = &handle->handles[ctx];
          CU_ERROR_CHECK(cuTaskCreate(&tasks[task], background_dgemm, &args, sizeof(struct dgemm_args)));
          CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, ctx++, tasks[task++]));
          if (ctx == nCtxs)
            ctx = 0;
        }
      }
    }
  }

  CUresult result;
  for (task = 0; task < nTasks; task++)
    CU_ERROR_CHECK(cuTaskDestroy(tasks[task], &result));

  return result;
}
