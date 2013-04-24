#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "ref/ctrsm_ref.c"
#include "util/clatmc.c"

int main(int argc, char * argv[]) {
  CBlasSide side;
  CBlasUplo uplo;
  CBlasTranspose trans;
  CBlasDiag diag;
  size_t m, n;

  if (argc != 7) {
    fprintf(stderr, "Usage: %s <side> <uplo> <trans> <diag> <m> <n>\n"
                    "where:\n"
                    "  side     is 'l' or 'L' for CBlasLeft and 'r' or 'R' for CBlasRight\n"
                    "  uplo     is 'u' or 'U' for CBlasUpper and 'l' or 'L' for CBlasLower\n"
                    "  trans    is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n"
                    "  diag     is 'n' or 'N' for CBlasNonUnit and 'u' or 'U' for CBlasUnit\n"
                    "  m and n  are the sizes of the matrices\n", argv[0]);
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

  char d;
  if (sscanf(argv[4], "%c", &d) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[4]);
    return 4;
  }
  switch (d) {
    case 'N': case 'n': diag = CBlasNonUnit; break;
    case 'U': case 'u': diag = CBlasUnit; break;
    default: fprintf(stderr, "Unknown diag '%c'\n", t); return 4;
  }

  if (sscanf(argv[5], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  if (sscanf(argv[6], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[6]);
    return 6;
  }

  srand(0);

  float complex alpha, * A, * B, * refB;
  size_t lda, ldb, * F, * G;

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

  alpha = ((float)rand() / (float)RAND_MAX) + ((float)rand() / (float)RAND_MAX) * I;

  if (side == CBlasLeft) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(float complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    if (clatmc(m, 2.0f, A, lda) != 0) {
      fputs("Unable to initialise A\n", stderr);
      return -1;
    }
  }
  else {
    lda = (n + 1u) & ~1u;
    if ((A = malloc(lda * n * sizeof(float complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    if (clatmc(n, 2.0f, A, lda) != 0) {
      fputs("Unable to initialise A\n", stderr);
      return -1;
    }
  }

  ldb = (m + 1u) & ~1u;
  if ((B = malloc(ldb * n * sizeof(float complex))) == NULL) {
    fputs("Unable to allocate B\n", stderr);
    return -3;
  }
  if ((refB = malloc(ldb * n * sizeof(float complex))) == NULL) {
    fputs("Unable to allocate refB\n", stderr);
    return -4;
  }
  if ((F = calloc(ldb * n, sizeof(float complex))) == NULL) {
    fputs("Unable to allocate F\n", stderr);
    return -5;
  }
  if ((G = calloc(ldb * n, sizeof(float complex))) == NULL) {
    fputs("Unable to allocate G\n", stderr);
    return -6;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refB[j * ldb + i] = B[j * ldb + i] = ((float)rand() / (float)RAND_MAX) + ((float)rand() / (float)RAND_MAX) * I;
  }

  ctrsm_ref(side, uplo, trans, diag, m, n, alpha, A, lda, refB, ldb, F, G);
  CU_ERROR_CHECK(cuMultiGPUCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));

  bool passed = true;
  float rdiff = 0.0f, idiff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      float d = fabsf(crealf(B[j * ldb + i]) - crealf(refB[j * ldb + i]));
      if (d > rdiff)
        rdiff = d;

      if (passed) {
        if (d > (float)F[j * ldb + i] * 2.0f * FLT_EPSILON)
          passed = false;
      }

      float c = fabsf(cimagf(B[j * ldb + i]) - cimagf(refB[j * ldb + i]));
      if (c > idiff)
        idiff = c;

      if (passed) {
        if (c > (float)G[j * ldb + i] * 2.0f * FLT_EPSILON)
          passed = false;
      }
    }
  }
  free(F);
  free(G);

  struct timespec start, stop;
  if (clock_gettime(CLOCK_REALTIME, &start) != 0) {
    fputs("clock_gettime failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));
  if (clock_gettime(CLOCK_REALTIME, &stop) != 0) {
    fputs("clock_gettime failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_nsec - start.tv_nsec) * 1.e-9) / 20.0;

  const size_t flops = (side == CBlasLeft) ?
                        (6 * (n * m * (m + 1) / 2) + 2 * (n * m * (m - 1) / 2)) :
                        (6 * (m * n * (n + 1) / 2) + 2 * (m * n * (n - 1) / 2));

  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(refB);

  CU_ERROR_CHECK(cuMultiGPUBLASDestroy(handle));
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return (int)!passed;
}
