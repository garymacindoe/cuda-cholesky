#include "lapack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "ref/strtri_ref.c"
#include "util/slatmc.c"

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  CBlasDiag diag;
  size_t n;

  if (argc != 4) {
    fprintf(stderr, "Usage: %s <uplo> <diag> <n>\nwhere:\n"
                    "  uplo  is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n"
                    "  diag  is 'u' or 'U' for CBlasUnit or 'n' or 'N' for CBlasNonUnit\n"
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

  char d;
  if (sscanf(argv[2], "%c", &d) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (d) {
    case 'U': case 'u': diag = CBlasUnit; break;
    case 'N': case 'n': diag = CBlasNonUnit; break;
    default: fprintf(stderr, "Unknown diag '%c'\n", d); return 1;
  }

  if (sscanf(argv[3], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
    return 3;
  }

  srand(0);

  float * A, * refA;
  size_t lda;
  long info, rInfo;

  lda = (n + 3u) & ~3u;
  if ((A = malloc(lda *  n * sizeof(float))) == NULL) {
    fputs("Unable to allocate A\n", stderr);
    return -1;
  }

  if ((refA = malloc(lda * n * sizeof(float))) == NULL) {
    fputs("Unable to allocate refA\n", stderr);
    return -2;
  }

  if (slatmc(n, 2.0f, A, lda) != 0) {
    fputs("Unable to initialise A\n", stderr);
    return -1;
  }

//   spotrf(uplo, n, A, lda, &info);
//   if (info != 0) {
//     fputs("Failed to compute Cholesky decomposition of A\n", stderr);
//     return (int)info;
//   }

  for (size_t j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(float));

  strtri_ref(uplo, diag, n, refA, lda, &rInfo);
  strtri(uplo, diag, n, A, lda, &info);

  bool passed = (info == rInfo);
  float diff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      float d = fabsf(A[j * lda + i] - refA[j * lda + i]);
      if (d > diff)
        diff = d;
    }
  }

  // Set A to identity so that repeated applications of the inverse
  // while benchmarking do not exit early due to singularity.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? 1.0f : 0.0f;
  }

  struct timespec start, stop;
  if (clock_gettime(CLOCK_REALTIME, &start) != 0) {
    fprintf(stderr, "clock_gettime failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    strtri(uplo, diag, n, A, lda, &info);
  if (clock_gettime(CLOCK_REALTIME, &stop) != 0) {
    fprintf(stderr, "clock_gettime failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_nsec - start.tv_nsec) * 1.e-6) / 20.0;
  const size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);

  return (int)!passed;
}
