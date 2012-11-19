#include "blas.h"
#include "error.h"
#include "cuhandle.h"
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include "sgemm_ref.c"

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

  float alpha, beta, * A, * B, * C, * refC;
  CUdeviceptr dA, dB, dC, dD;
  size_t lda, ldb, ldc, dlda, dldb, dldc, dldd;

  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUhandle handle;
  CU_ERROR_CHECK(cuHandleCreate(&handle, CU_CTX_BLOCKING_SYNC, device));

  alpha = (float)rand() / (float)RAND_MAX;
  beta = (float)rand() / (float)RAND_MAX;

  if (transA == CBlasNoTrans) {
    lda = (m + 3u) & ~3u;
    if ((A = malloc(lda * k * sizeof(float))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, m * sizeof(float), k, sizeof(float)));
    dlda /= sizeof(float);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float),
                           m * sizeof(float), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    lda = (k + 3u) & ~3u;
    if ((A = malloc(lda * m * sizeof(float))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, k * sizeof(float), m, sizeof(float)));
    dlda /= sizeof(float);

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float),
                           k * sizeof(float), m };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  if (transB == CBlasNoTrans) {
    ldb = (k + 3u) & ~3u;
    if ((B = malloc(ldb * n * sizeof(float))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, k * sizeof(float), n, sizeof(float)));
    dldb /= sizeof(float);

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(float),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dB, NULL, dldb * sizeof(float),
                           k * sizeof(float), n };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    ldb = (n + 3u) & ~3u;
    if ((B = malloc(ldb * k * sizeof(float))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, n * sizeof(float), k, sizeof(float)));
    dldb /= sizeof(float);

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(float),
                           0, 0, CU_MEMORYTYPE_DEVICE, NULL, dB, NULL, dldb * sizeof(float),
                           n * sizeof(float), k };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  ldc = (m + 3u) & ~3u;
  if ((C = malloc(ldc * n * sizeof(float))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(float))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&dC, &dldc, m * sizeof(float), n, sizeof(float)));
  dldc /= sizeof(float);
  CU_ERROR_CHECK(cuMemAllocPitch(&dD, &dldd, m * sizeof(float), n, sizeof(float)));
  dldd /= sizeof(float);

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = (float)rand() / (float)RAND_MAX;
  }

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, C, 0, NULL, ldc * sizeof(float),
                         0, 0, CU_MEMORYTYPE_DEVICE, NULL, dC, NULL, dldc * sizeof(float),
                         m * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  sgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  CU_ERROR_CHECK(cuSgemm2(handle, transA, transB, m, n, k, alpha, dA, dlda, dB, dldb, beta, dC, dldc, dD, dldd, NULL));

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_DEVICE, NULL, dD, NULL, dldd * sizeof(float),
           0, 0, CU_MEMORYTYPE_HOST, C, 0, NULL, ldc * sizeof(float),
           m * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  float diff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      float d = fabsf(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }
  free(refC);

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuSgemm2(handle, transA, transB, m, n, k, alpha, dA, dlda, dB, dldb, beta, dC, dldc, dD, dldd, NULL));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  size_t flops = 2 * k - 1;
  if (alpha != 1.0f)
    flops += 1;
  if (beta != 0.0f)
    flops += 2;
  float error = (float)flops * 2.0f * FLT_EPSILON;
  flops *= m * n;

  bool passed = (diff <= error);
  fprintf(stdout, "%.3ems %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((float)flops * 1.e-6f) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dB));
  CU_ERROR_CHECK(cuMemFree(dC));
  CU_ERROR_CHECK(cuMemFree(dD));

  CU_ERROR_CHECK(cuHandleDestroy(handle));

  return (int)!passed;
}
