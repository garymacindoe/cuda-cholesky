#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include "dtrmm_ref.c"

int main(int argc, char * argv[]) {
  CBlasSide side;
  CBlasUplo uplo;
  CBlasTranspose trans;
  CBlasDiag diag;
  size_t m, n;
  int d = 0;

  if (argc < 7 || argc > 8) {
    fprintf(stderr, "Usage: %s <side> <uplo> <trans> <diag> <m> <n> [device]\n"
                    "where:\n"
                    "  side               is 'l' or 'L' for CBlasLeft and 'r' or 'R' for CBlasRight\n"
                    "  uplo               is 'u' or 'U' for CBlasUpper and 'l' or 'L' for CBlasLower\n"
                    "  trans              is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
                    "  diag               is 'n' or 'N' for CBlasNonUnit and 'u' or 'U' for CBlasUnit\n"
                    "  m and n           are the sizes of the matrices\n"
                    "  device             is the GPU to use (default 0)\n", argv[0]);
    return 1;
  }

  char s;
  if (sscanf(argv[1], "%c", &s) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (s) {
    case 'L': case 'l': side = CBlasLeft; break;
    case 'R': case 'r': side = CBlasRight; break;
    default: fprintf(stderr, "Unknown side '%c'\n", s); return 1;
  }

  char u;
  if (sscanf(argv[2], "%c", &u) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (u) {
    case 'U': case 'u': uplo = CBlasUpper; break;
    case 'L': case 'l': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 2;
  }

  char t;
  if (sscanf(argv[3], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[3]);
    return 3;
  }
  switch (t) {
    case 'N': case 'n': trans = CBlasNoTrans; break;
    case 'T': case 't': trans = CBlasTrans; break;
    case 'C': case 'c': trans = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 3;
  }

  char di;
  if (sscanf(argv[4], "%c", &di) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[4]);
    return 4;
  }
  switch (di) {
    case 'N': case 'n': diag = CBlasNonUnit; break;
    case 'U': case 'u': diag = CBlasUnit; break;
    default: fprintf(stderr, "Unknown diag '%c'\n", di); return 4;
  }

  if (sscanf(argv[5], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  if (sscanf(argv[6], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[6]);
    return 6;
  }

  if (argc > 7) {
    if (sscanf(argv[7], "%d", &d) != 1) {
      fprintf(stderr, "Unable to parse number from '%s'\n", argv[7]);
      return 7;
    }
  }

  srand(0);

  double alpha, * A, * B, * refB;
  CUdeviceptr dA, dB, dX;
  size_t lda, ldb, dlda, dldb, dldx;

  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device));

  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoad(&module, "dtrmm.fatbin"));

  alpha = (double)rand() / (double)RAND_MAX;

  if (side == CBlasLeft) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, m * sizeof(double), m, sizeof(double)));
    dlda /= sizeof(double);

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(double),
    0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(double),
    m * sizeof(double), m };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }
  else {
    lda = (n + 1u) & ~1u;
    if ((A = malloc(lda * n * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }
    CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(double), n, sizeof(double)));
    dlda /= sizeof(double);

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < n; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }

    CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(double),
    0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(double),
    n * sizeof(double), n };
    CU_ERROR_CHECK(cuMemcpy2D(&copy));
  }

  ldb = (m + 1u) & ~1u;
  if ((B = malloc(ldb * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate B\n", stderr);
    return -3;
  }
  if ((refB = malloc(ldb * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate refB\n", stderr);
    return -4;
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&dB, &dldb, m * sizeof(double), n, sizeof(double)));
  dldb /= sizeof(double);
  CU_ERROR_CHECK(cuMemAllocPitch(&dX, &dldx, m * sizeof(double), n, sizeof(double)));
  dldx /= sizeof(double);

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refB[j * ldb + i] = B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
  }

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(double),
  0, 0, CU_MEMORYTYPE_DEVICE, NULL, dB, NULL, dldb * sizeof(double),
  m * sizeof(double), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  dtrmm_ref(side, uplo, trans, diag, m, n, alpha, A, lda, refB, ldb);
  CU_ERROR_CHECK(cuDtrmm2(module, side, uplo, trans, diag, m, n, alpha, dA, dlda, dB, dldb, dX, dldx, NULL));

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_DEVICE, NULL, dX, NULL, dldx * sizeof(double),
  0, 0, CU_MEMORYTYPE_HOST, B, 0, NULL, ldb * sizeof(double),
  m * sizeof(double), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  bool passed = true;
  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(B[j * ldb + i] - refB[j * ldb + i]);
      if (d > diff)
        diff = d;

      size_t flops;
      if (side == CBlasLeft)
        flops = 2 * i + 1;
      else
        flops = 2 * j + 1;
      if (diag == CBlasNonUnit)
        flops++;

      if (d > (double)flops * 2.0 * DBL_EPSILON)
        passed = false;
    }
  }

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuDtrmm2(module, side, uplo, trans, diag, m, n, alpha, dA, dlda, dB, dldb, dX, dldx, NULL));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  size_t flops = m * n;
  if (alpha != 0.0) {
    flops += (side == CBlasLeft) ? m * m * n : m * n * n;
    if (diag == CBlasNonUnit) flops += m * n;
  }

  fprintf(stdout, "%.3ems %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((float)flops * 1.e-6f) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(refB);
  CU_ERROR_CHECK(cuMemFree(dA));
  CU_ERROR_CHECK(cuMemFree(dB));
  CU_ERROR_CHECK(cuMemFree(dX));

  CU_ERROR_CHECK(cuModuleUnload(module));

  CU_ERROR_CHECK(cuCtxDestroy(context));

  return (int)!passed;
}
