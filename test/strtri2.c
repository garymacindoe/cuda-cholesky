#include "lapack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include "strtri_ref.c"

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
    default: fprintf(stderr, "Unknown uplo '%c'\n", d); return 1;
  }

  if (sscanf(argv[3], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
    return 3;
  }

  srand(0);

  float * A, * B, * refB, * C;
  size_t lda, ldb, ldc, k = 5 * n;
  long info, rInfo;

  lda = (n + 3u) & ~3u;
  if ((A = malloc(lda *  n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return -1;
  }

  ldb = (n + 3u) & ~3u;
  if ((B = malloc(ldb *  n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate B\n");
    return -2;
  }

  if ((refB = malloc(ldb * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate refB\n");
    return -3;
  }

  ldc = (k + 3u) & ~3u;
  if ((C = malloc(ldc * n * sizeof(float))) == NULL) {
    fprintf(stderr, "Unable to allocate C\n");
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < k; i++)
      C[j * ldc + i] = gaussian();
  }
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      float temp = 0.0f;
      for (size_t l = 0; l < k; l++)
        temp += C[i * ldc + l] * C[j * ldc + l];
      A[j * lda + i] = 0.01f * temp;
    }
  }
  free(C);

  spotrf(uplo, n, A, lda, &info);
  if (info != 0) {
    fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
    return (int)info;
  }

  for (size_t j = 0; j < n; j++)
    memcpy(&refB[j * ldb], &A[j * lda], n * sizeof(float));

  strtri_ref(uplo, diag, n, refB, ldb, &rInfo);
  strtri2(uplo, diag, n, A, lda, B, ldb, &info);

  bool passed = (info == rInfo);
  float diff = 0.0f;
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < j; i++) {
        float d = fabsf(B[j * ldb + i] - refB[j * ldb + i]);
        if (d > diff)
          diff = d;
      }
      if (diag == CBlasNonUnit) {
        float d = fabsf(B[j * ldb + j] - refB[j * ldb + j]);
        if (d > diff)
          diff = d;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      if (diag == CBlasNonUnit) {
        float d = fabsf(B[j * ldb + j] - refB[j * ldb + j]);
        if (d > diff)
          diff = d;
      }
      for (size_t i = j + 1; i < n; i++) {
        float d = fabsf(B[j * ldb + i] - refB[j * ldb + i]);
        if (d > diff)
          diff = d;
      }
    }
  }

  // Set A to identity so that repeated applications of the inverse
  // while benchmarking do not exit early due to singularity.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? 1.0f : 0.0f;
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    strtri2(uplo, diag, n, A, lda, B, ldb, &info);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(refB);

  return (int)!passed;
}
