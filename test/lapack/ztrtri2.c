#include "lapack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <complex.h>
#include <sys/time.h>
#include "ref/ztrtri_ref.c"
#include "util/zlatmc.c"

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

  double complex * A, * B, * refB;
  size_t lda, ldb;
  long info, rInfo;

  lda = n;
  if ((A = malloc(lda *  n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate A\n", stderr);
    return -1;
  }

  ldb = n;
  if ((B = malloc(ldb *  n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate B\n", stderr);
    return -2;
  }

  if ((refB = malloc(ldb * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate refB\n", stderr);
    return -3;
  }

  if (zlatmc(n, 2.0, A, lda) != 0) {
    fputs("Unable to initialise A\n", stderr);
    return -1;
  }

//   zpotrf(uplo, n, A, lda, &info);
//   if (info != 0) {
//     fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
//     return (int)info;
//   }

  for (size_t j = 0; j < n; j++)
    memcpy(&refB[j * ldb], &A[j * lda], n * sizeof(double complex));

  ztrtri_ref(uplo, diag, n, refB, ldb, &rInfo);
  ztrtri2(uplo, diag, n, A, lda, B, ldb, &info);

  bool passed = (info == rInfo);
  double rdiff = 0.0, idiff = 0.0;
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < j; i++) {
        double diff = fabs(creal(refB[j * ldb + i]) - creal(B[j * ldb + i]));
        if (diff > rdiff)
          rdiff = diff;
        diff = fabs(cimag(refB[j * ldb + i]) - cimag(B[j * ldb + i]));
        if (diff > idiff)
          idiff = diff;
      }
      if (diag == CBlasNonUnit) {
        double diff = fabs(creal(refB[j * ldb + j]) - creal(B[j * ldb + j]));
        if (diff > rdiff)
          rdiff = diff;
        diff = fabs(cimag(refB[j * ldb + j]) - cimag(B[j * ldb + j]));
        if (diff > idiff)
          idiff = diff;
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      if (diag == CBlasNonUnit) {
        double diff = fabs(creal(refB[j * ldb + j]) - creal(B[j * ldb + j]));
        if (diff > rdiff)
          rdiff = diff;
        diff = fabs(cimag(refB[j * ldb + j]) - cimag(B[j * ldb + j]));
        if (diff > idiff)
          idiff = diff;
      }
      for (size_t i = j + 1; i < n; i++) {
        double diff = fabs(creal(refB[j * ldb + i]) - creal(B[j * ldb + i]));
        if (diff > rdiff)
          rdiff = diff;
        diff = fabs(cimag(refB[j * ldb + i]) - cimag(B[j * ldb + i]));
        if (diff > idiff)
          idiff = diff;
      }
    }
  }

  // Set A to identity so that repeated applications of the inverse
  // while benchmarking do not exit early due to singularity.
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = (i == j) ? (1.0 + 0.0 * I) : (0.0 + 0.0 * I);
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    ztrtri2(uplo, diag, n, A, lda, B, ldb, &info);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(refB);

  return (int)!passed;
}
