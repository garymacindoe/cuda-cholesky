#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include "slogdet.fatbin.c"

static inline unsigned int max(unsigned int a, unsigned int b) { return (a > b) ? a : b; }

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

CUresult cuSlogdet(CUdeviceptr x, size_t incx, size_t n, float * result, CUstream stream) {
  if (n == 0) {
    *result = 0.0f;
    return CUDA_SUCCESS;
  }

  static CUmodule module = NULL;
  if (module == NULL)
    CU_ERROR_CHECK(cuModuleLoadData(&module, imageBytes));

  unsigned int threads, blocks;
  if (n == 1) {
    threads = 1;
    blocks = 1;
  }
  else {
    threads = (n < 1024) ? nextPow2((unsigned int )(n / 2)) : 512;
    blocks = max(1, (unsigned int)n / (threads * 2));
  }

  CUdeviceptr temp;
  CU_ERROR_CHECK(cuMemAlloc(&temp, blocks * sizeof(float)));

  char name[31];
  snprintf(name, 31, "_Z6reduceILj%uELb%dEEvPKfPfii", threads, (n & (n - 1)) == 0);

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, name));

  void * params[] = { &x, &temp, &incx, &n };

  CU_ERROR_CHECK(cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, stream, params, NULL));

  CU_ERROR_CHECK(cuMemcpyDtoHAsync(result, temp, sizeof(float), stream));

  CU_ERROR_CHECK(cuMemFree(temp));

  return CUDA_SUCCESS;
}
