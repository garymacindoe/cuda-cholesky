#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <sys/time.h>
#include "ref/cpotri_ref.c"
#include "util/clatmc.c"

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  size_t n;

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <uplo> <diag> <n>\nwhere:\n"
                    "  uplo  is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n"
                    "  n     is the size of the matrix\n", argv[0]);
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

  float complex * A, * refA;
  size_t lda;
  long info, rInfo;

  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUdevice devices[deviceCount];
  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  CUmultiGPU mGPU;
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, devices, deviceCount));

  CUmultiGPULAPACKhandle handle;
  CU_ERROR_CHECK(cuMultiGPULAPACKCreate(&handle, mGPU));

  lda = (n + 1u) & ~1u;
  if ((A = malloc(lda *  n * sizeof(float complex))) == NULL) {
    fputs("Unable to allocate A\n", stderr);
    return -1;
  }

  if ((refA = malloc(lda * n * sizeof(float complex))) == NULL) {
    fputs("Unable to allocate refA\n", stderr);
    return -2;
  }

  if (clatmc(n, 2.0f, A, lda) != 0) {
    fputs("Unable to initialise A\n", stderr);
    return -1;
  }

//   cpotrf(uplo, n, A, lda, &info);
//   if (info != 0) {
//     fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
//     return (int)info;
//   }

  for (size_t j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(float complex));

  cpotri_ref(uplo, n, refA, lda, &rInfo);
  CU_ERROR_CHECK(cuMultiGPUCpotri(handle, uplo, n, A, lda, &info));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));

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

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUCpotri(handle, uplo, n, A, lda, &info));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0f;
  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);

  CU_ERROR_CHECK(cuMultiGPULAPACKDestroy(handle));
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return (int)!passed;
}
