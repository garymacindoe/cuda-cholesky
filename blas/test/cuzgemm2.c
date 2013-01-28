#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include "ref/zgemm_ref.c"

int main(int argc, char * argv[]) {
  CBlasTranspose transA, transB;
  size_t m, n, k;
  int d = 0;

  if (argc < 6 || argc > 7) {
    fprintf(stderr, "Usage: %s <transA> <transB> <m> <n> <k> [device]\n"
                    "where:\n"
                    "  transA and transB  are 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
                    "  m, n and k         are the sizes of the matrices\n"
                    "  device             is the GPU to use (default 0)\n", argv[0]);
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

  if (argc > 6) {
    if (sscanf(argv[6], "%d", &d) != 1) {
      fprintf(stderr, "Unable to parse number from '%s'\n", argv[6]);
      return 6;
    }
  }

  srand(0);

  double complex alpha, beta, * A, * B, * C, * refC;
  CUdeviceptr dA, dB, dC, dD;
  size_t lda, ldb, ldc, dlda, dldb, dldc, dldd;

  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC, device));

  CUblashandle handle;
  CU_ERROR_CHECK(cuBlasHandleCreate(&handle));

  alpha = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
  beta = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;

  if (transA == CBlasNoTrans) {
    lda = m;
    if ((A = malloc(lda * k * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, m * sizeof(double complex), k, sizeof(double complex)));
    dlda /= sizeof(double complex);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(double complex),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(double complex),
                           m * sizeof(double complex), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    lda = k;
    if ((A = malloc(lda * m * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, k * sizeof(double complex), m, sizeof(double complex)));
    dlda /= sizeof(double complex);

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(double complex),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(double complex),
                           k * sizeof(double complex), m };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  if (transB == CBlasNoTrans) {
    ldb = k;
    if ((B = malloc(ldb * n * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, k * sizeof(double complex), n, sizeof(double complex)));
    dldb /= sizeof(double complex);

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(double complex),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dB, NULL, dldb * sizeof(double complex),
                           k * sizeof(double complex), n };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    ldb = n;
    if ((B = malloc(ldb * k * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, n * sizeof(double complex), k, sizeof(double complex)));
    dldb /= sizeof(double complex);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(double complex),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dB, NULL, dldb * sizeof(double complex),
                           n * sizeof(double complex), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  ldc = m;
  if ((C = malloc(ldc * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&dC, &dldc, m * sizeof(double complex), n, sizeof(double complex)));
  dldc /= sizeof(double complex);
  CU_ERROR_CHECK(cuMemAllocPitch(&dD, &dldd, m * sizeof(double complex), n, sizeof(double complex)));
  dldd /= sizeof(double complex);

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
  }

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, C, 0, NULL, ldc * sizeof(double complex),
                         0, 0, CU_MEMORYTYPE_DEVICE, NULL, dC, NULL, dldc * sizeof(double complex),
                         m * sizeof(double complex), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  zgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  CU_ERROR_CHECK(cuZgemm2(handle, transA, transB, m, n, k, alpha, dA, dlda, dB, dldb, beta, dC, dldc, dD, dldd, NULL));

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_DEVICE, NULL, dD, NULL, dldd * sizeof(double complex),
           0, 0, CU_MEMORYTYPE_HOST, C, 0, NULL, ldc * sizeof(double complex),
           m * sizeof(double complex), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  double rdiff = 0.0, idiff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(creal(C[j * ldc + i]) - creal(refC[j * ldc + i]));
      if (d > rdiff)
        rdiff = d;
      d = fabs(cimag(C[j * ldc + i]) - cimag(refC[j * ldc + i]));
      if (d > idiff)
        idiff = d;
    }
  }

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuZgemm2(handle, transA, transB, m, n, k, alpha, dA, dlda, dB, dldb, beta, dC, dldc, dD, dldd, NULL));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  size_t flops = 8 * k - 2;
  if (alpha != 1.0 + 0.0 * I)
    flops += 6;
  if (beta != 0.0 + 0.0 * I)
    flops += 8;
  double error = (double)flops * DBL_EPSILON;
  flops *= m * n;

  bool passed = (rdiff <= error) && (idiff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time * 1.e-3f,
          ((float)flops * 1.e-6f) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);
  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dB));
  CU_ERROR_CHECK(cuMemFree(dC));
  CU_ERROR_CHECK(cuMemFree(dD));

  CU_ERROR_CHECK(cuBlasHandleDestroy(handle));

  CU_ERROR_CHECK(cuCtxDestroy(context));

  return (int)!passed;
}
