#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "dgemm_ref.c"

static CUresult createContext(const void * args) {
  CUdevice * device = (CUdevice *)args;
  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_YIELD, *device));
  return CUDA_SUCCESS;
}

static CUresult destroyContext(const void * args) {
  (void)args;

  CUcontext context;
  CU_ERROR_CHECK(cuCtxGetCurrent(&context));
  CU_ERROR_CHECK(cuCtxDestroy(context));

  return CUDA_SUCCESS;
}

int main(int argc, char * argv[]) {
  CBlasTranspose transA, transB;
  size_t m, n, k;

  if (argc != 6) {
    fprintf(stderr, "Usage: %s <transA> <transB> <m> <n> <k>\n"
                    "where:\n"
                    "  transA and transB  are 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
                    "  m, n and k         are the sizes of the matrices\n", argv[0]);
    return 1;
  }

  char t;
  if (sscanf(argv[1], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (t) {
    case 'N': case 'n': transA = CBlasNoTrans; break;
    case 'T': case 't': transA = CBlasTrans; break;
    case 'C': case 'c': transA = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[2], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (t) {
    case 'N': case 'n': transB = CBlasNoTrans; break;
    case 'T': case 't': transB = CBlasTrans; break;
    case 'C': case 'c': transB = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[3], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
    return 3;
  }

  if (sscanf(argv[4], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[4]);
    return 4;
  }

  if (sscanf(argv[5], "%zu", &k) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  srand(0);
  double alpha, beta, * A, * B, * C, * refC;
  size_t lda, ldb, ldc;

  CU_ERROR_CHECK(cuInit(0));

  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUthread threads[deviceCount];
  for (int i = 0; i < deviceCount; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, createContext, &device, sizeof(CUdevice)));

    CU_ERROR_CHECK(cuThreadCreate(&threads[i]));
    CU_ERROR_CHECK(cuThreadRunTask(threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return (int)result;
  }

  alpha = (double)rand() / (double)RAND_MAX;
  beta = (double)rand() / (double)RAND_MAX;

  if (transA == CBlasNoTrans) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * k * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }
  else {
    lda = (k + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }

  if (transB == CBlasNoTrans) {
    ldb = (k + 1u) & ~1u;
    if ((B = malloc(ldb * n * sizeof(double))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
    }
  }
  else {
    ldb = (n + 1u) & ~1u;
    if ((B = malloc(ldb * k * sizeof(double))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
    }
  }

  ldc = (m + 1u) & ~1u;
  if ((C = malloc(ldc * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = (double)rand() / (double)RAND_MAX;
  }

  dgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  CU_ERROR_CHECK(cuMultiGPUDgemm(threads, deviceCount, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));

  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUDgemm(threads, deviceCount, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = 2 * k - 1;
  if (alpha != 1.0)
    flops += 1;
  if (beta != 0.0)
    flops += 2;
  double error = (double)flops * 2.0 * DBL_EPSILON;
  flops *= m * n;

  bool passed = (diff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);

  for (int i = 0; i < deviceCount; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, destroyContext, NULL, 0));
    CU_ERROR_CHECK(cuThreadRunTask(threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != 0)
      return result;

    CU_ERROR_CHECK(cuThreadDestroy(threads[i]));
  }

  return (int)!passed;
}
