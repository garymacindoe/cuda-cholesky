#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "handle.h"
#include "slogdet.fatbin.c"

static inline unsigned int max(unsigned int a, unsigned int b) { return (a > b) ? a : b; }
static inline unsigned int min(unsigned int a, unsigned int b) { return (a < b) ? a : b; }

static inline unsigned int nextPow2(unsigned int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}

float slogdet(const float * x, size_t incx, size_t n) {
  if (n == 0)
    return 0.0f;

  float total = 0.0f;
  if (incx == 1) {
    for (size_t i = 0; i < n; i++)
      total += logf(x[i]);      // SSE with a vector math library (for log)
  }
  else {
    for (size_t i = 0; i < n; i++)
      total += logf(x[i * incx]);
  }

  return 2.0f * total;
}

CUresult cuSlogdet(CULAPACKhandle handle, CUdeviceptr x, size_t incx, size_t n, CUdeviceptr work, size_t * lwork, CUstream stream) {
  if (n == 0)
    return CUDA_SUCCESS;

  unsigned int threads, blocks;
  if (n == 1) {
    threads = 1;
    blocks = 1;
  }
  else {
    threads = (n < 256) ? nextPow2((unsigned int )(n / 2)) : 128;
    blocks = max(1, (unsigned int)n / (threads * 2));
  }

  blocks = min(64, blocks);

  if (work == 0) {
    *lwork = blocks;
    return CUDA_SUCCESS;
  }
  else if (*lwork < blocks)
    return CUDA_ERROR_INVALID_VALUE;

  if (handle->slogdet == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&handle->slogdet, imageBytes));

  char name[31];
  snprintf(name, 31, "_Z6reduceILj%uELb%dEEvPKfPfii", threads, (n & (n - 1)) == 0);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, handle->slogdet, name));

  void * params[] = { &x, &work, &incx, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}
