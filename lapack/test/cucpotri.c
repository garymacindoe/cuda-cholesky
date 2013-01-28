#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cpotri_ref.c"

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  size_t n;
  int d = 0;

  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: %s <uplo> <n>\n"
                    "where:\n"
                    "  uplo    is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n"
                    "  n       is the size of the matrix\n"
                    "  device  is the GPU to use (default 0)\n", argv[0]);
    return 1;
  }

  char u;
  if (sscanf(argv[1], "%c", &u) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (u) {
    case 'U': case 'u': uplo = CBlasUpper; break;
    case 'L': case 'l': uplo = CBlasLower; break;
    default: fprintf(stderr, "Unknown uplo '%c'\n", u); return 1;
  }

  if (sscanf(argv[2], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[2]);
    return 2;
  }

  srand(0);

  float complex * A, * C, * refA;
  CUdeviceptr dA;
  size_t lda, ldc, dlda, k = 5 * n;
  long info, rInfo;

  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC, device));

  lda = (n + 1u) & ~1u;
  if ((A = malloc(lda *  n * sizeof(float complex))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return -1;
  }
  if ((refA = malloc(lda * n * sizeof(float complex))) == NULL) {
    fprintf(stderr, "Unable to allocate refA\n");
    return -2;
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(float complex), n, sizeof(float complex)));
  dlda /= sizeof(float complex);

  ldc = (k + 1u) & ~1u;
  if ((C = malloc(ldc * n * sizeof(float complex))) == NULL) {
    fprintf(stderr, "Unable to allocate C\n");
    return -3;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < k; i++)
      C[j * ldc + i] = gaussian();
  }
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      float complex temp = 0.0f + 0.0f * I;
      for (size_t l = 0; l < k; l++)
        temp += C[i * ldc + l] * C[j * ldc + l];
      refA[j * lda + i] = A[j * lda + i] = temp;
    }
  }
  free(C);

  cpotrf(uplo, n, A, lda, &info);
  if (info != 0) {
    fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
    return (int)info;
  }

  for (size_t j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(float complex));

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float complex),
                         0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float complex),
                         n * sizeof(float complex), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  cpotri_ref(uplo, n, refA, lda, &rInfo);
  CU_ERROR_CHECK(cuCpotri(uplo, n, dA, dlda, &info));

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float complex),
                          0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float complex),
                          n * sizeof(float complex), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  bool passed = (info == rInfo);
  float rdiff = 0.0f, idiff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      float d = fabsf(crealf(A[j * lda + i]) - crealf(refA[j * lda + i]));
      if (d > rdiff)
        rdiff = d;
      d = fabsf(cimagf(A[j * lda + i]) - cimagf(refA[j * lda + i]));
      if (d > idiff)
        idiff = d;
    }
  }

  // Set A to identity so that repeated applications of the inverse
  // while benchmarking do not exit early due to singularity.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? (1.0f + 0.0f * I) : (0.0f + 0.0f * I);
  }

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float complex),
                          0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float complex),
                          n * sizeof(float complex), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuCpotri(uplo, n, dA, dlda, &info));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((float)flops * 1.e-6f) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);
  CU_ERROR_CHECK(cuMemFree(dA));

  CU_ERROR_CHECK(cuCtxDestroy(context));

  return (int)!passed;
}
