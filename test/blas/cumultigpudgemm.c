#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "ref/dgemm_ref.c"

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

  CUdevice devices[deviceCount];
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  CUmultiGPU mGPU;
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, devices, deviceCount));

  CUmultiGPUBLAShandle handle;
  CU_ERROR_CHECK(cuMultiGPUBLASCreate(&handle, mGPU));

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
  CU_ERROR_CHECK(cuMultiGPUDgemm(handle, transA, transB, m, n, k,
                                 alpha, A, lda, B, ldb, beta, C, ldc));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));

  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }

  struct timespec start, stop;
  if (clock_gettime(CLOCK_REALTIME, &start) != 0) {
    fputs("clock_gettime failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUDgemm(handle, transA, transB, m, n, k,
                                   alpha, A, lda, B, ldb, beta, C, ldc));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));
  if (clock_gettime(CLOCK_REALTIME, &stop) != 0) {
    fputs("clock_gettime failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_nsec - start.tv_nsec) * 1.e-9) / 20.0;

  size_t flops = 2 * k - 1;     // k multiplies and k - 1 adds per element
  if (alpha != 1.0)
    flops += 1;                 // additional multiply by alpha
  if (beta != 0.0)
    flops += 2;                 // additional multiply and add by beta
  double error = (double)flops * 2.0 * DBL_EPSILON;   // maximum per element error
  flops *= m * n;               // m * n elements

  bool passed = (diff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);

  CU_ERROR_CHECK(cuMultiGPUBLASDestroy(handle));
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return (int)!passed;
}
