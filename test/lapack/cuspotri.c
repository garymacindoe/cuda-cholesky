#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "ref/spotri_ref.c"

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

  float * A, * refA;//, * C;
  CUdeviceptr dA;
  size_t lda, dlda;//, ldc, k = 5 * n;
  long info, rInfo;

  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, d));

  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_BLOCKING_SYNC, device));

  lda = (n + 3u) & ~3u;
  if ((A = malloc(lda *  n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return -1;
  }
  if ((refA = malloc(lda * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate refA\n");
    return -2;
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&dA, &dlda, n * sizeof(float), n, sizeof(float)));
  dlda /= sizeof(float);

//   ldc = (k + 3u) & ~3u;
//   if ((C = malloc(ldc * n * sizeof(float))) == NULL) {
//     fprintf(stderr, "Unable to allocate C\n");
//     return -3;
//   }

//   for (size_t j = 0; j < n; j++) {
//     for (size_t i = 0; i < k; i++)
//       C[j * ldc + i] = gaussian();
//   }
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
//       float temp = 0.0f;
//       for (size_t l = 0; l < k; l++)
//         temp += C[i * ldc + l] * C[j * ldc + l];
      refA[j * lda + i] = A[j * lda + i] = gaussian();//temp;
    }
  }
//   free(C);

//   spotrf(uplo, n, A, lda, &info);
//   if (info != 0) {
//     fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
//     return (int)info;
//   }

//   for (size_t j = 0; j < n; j++)
//     memcpy(&refA[j * lda], &A[j * lda], n * sizeof(float));

  CUDA_MEMCPY2D copy = { 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float),
                         0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float),
                         n * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  spotri_ref(uplo, n, refA, lda, &rInfo);
  CU_ERROR_CHECK(cuSpotri(uplo, n, dA, dlda, &info));

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float),
                          0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float),
                          n * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  bool passed = (info == rInfo);
  float diff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      float d = fabsf(A[j * lda + i] - refA[j * lda + i]);
      if (d > diff)
        diff = d;
    }
  }

  // Set A to identity so that repeated applications of the cholesky
  // decomposition while benchmarking do not exit early due to
  // non-positive-definite-ness.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? 1.0f : 0.0f;
  }

  copy = (CUDA_MEMCPY2D){ 0, 0, CU_MEMORYTYPE_HOST, A, 0, NULL, lda * sizeof(float),
                          0, 0, CU_MEMORYTYPE_DEVICE, NULL, dA, NULL, dlda * sizeof(float),
                          n * sizeof(float), n };
  CU_ERROR_CHECK(cuMemcpy2D(&copy));

  CUevent start, stop;
  CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_BLOCKING_SYNC));
  CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_BLOCKING_SYNC));

  CU_ERROR_CHECK(cuEventRecord(start, NULL));
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuSpotri(uplo, n, dA, dlda, &info));
  CU_ERROR_CHECK(cuEventRecord(stop, NULL));
  CU_ERROR_CHECK(cuEventSynchronize(stop));

  float time;
  CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
  time /= 20;

  CU_ERROR_CHECK(cuEventDestroy(start));
  CU_ERROR_CHECK(cuEventDestroy(stop));

  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((float)flops * 1.e-6f) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);
  CU_ERROR_CHECK(cuMemFree(dA));

  CU_ERROR_CHECK(cuCtxDestroy(context));

  return (int)!passed;
}
