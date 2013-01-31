#include "lapack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include "ref/dpotrf_ref.c"

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  size_t n;

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <uplo> <n>\n"
                    "where:\n"
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

  double * A, * refA, * C;
  size_t lda, ldc, k = 5 * n;
  long info, rInfo;

  lda = (n + 1u) & ~1u;
  if ((A = malloc(lda *  n * sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return -1;
  }
  if ((refA = malloc(lda * n * sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate refA\n");
    return -2;
  }

  ldc = (k + 1u) & ~1u;
  if ((C = malloc(ldc * n * sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate C\n");
    return -3;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < k; i++)
      C[j * ldc + i] = gaussian();
  }
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      double temp = 0.0;
      for (size_t l = 0; l < k; l++)
        temp += C[i * ldc + l] * C[j * ldc + l];
      refA[j * lda + i] = A[j * lda + i] = temp;
    }
  }
  free(C);

  dpotrf_ref(uplo, n, refA, lda, &rInfo);
  dpotrf(uplo, n, A, lda, &info);

  bool passed = (info == rInfo);
  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      double d = fabs(A[j * lda + i] - refA[j * lda + i]);
      if (d > diff)
        diff = d;
    }
  }

  // Set A to identity so that repeated applications of the cholesky
  // decomposition while benchmarking do not exit early due to
  // non-positive-definite-ness.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? 1.0 : 0.0;
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    dpotrf(uplo, n, A, lda, &info);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  size_t flops = ((n * n * n) / 3) + ((n * n) / 2) + (n / 6);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);

  return (int)!passed;
}
