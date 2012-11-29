#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <sys/time.h>
#include "ztrmm_ref.c"

// extern void ztrmm_(const char *, const char *, const char *, const char *,
//                    const size_t *, const size_t *,
//                    const double complex *, const double complex *, const size_t *,
//                    const double complex *, const size_t *);

int main(int argc, char * argv[]) {
  CBlasSide side;
  CBlasUplo uplo;
  CBlasTranspose trans;
  CBlasDiag diag;
  size_t m, n;

  if (argc != 7) {
    fprintf(stderr, "Usage: %s <side> <uplo> <trans> <diag> <m> <n>\nwhere:\n  side               is 'l' or 'L' for CBlasLeft and 'r' or 'R' for CBlasRight\n  uplo               is 'u' or 'U' for CBlasUpper and 'l' or 'L' for CBlasLower\n  trans              is 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n  diag               is 'n' or 'N' for CBlasNonUnit and 'u' or 'U' for CBlasUnit\n  m and n           are the sizes of the matrices\n", argv[0]);
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

  double complex alpha, * A, * B, * refB, * C;
  size_t lda, ldb, ldc;

  alpha = (double)rand() / (double)RAND_MAX + ((double)rand() / (double)RAND_MAX) * I;

  if (side == CBlasLeft) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX + ((double)rand() / (double)RAND_MAX) * I;
    }
  }
  else {
    lda = (n + 1u) & ~1u;
    if ((A = malloc(lda * n * sizeof(double complex))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < n; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX + ((double)rand() / (double)RAND_MAX) * I;
    }
  }

  ldb = (m + 1u) & ~1u;
  if ((B = malloc(ldb * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate B\n", stderr);
    return -3;
  }
  if ((refB = malloc(ldb * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate refB\n", stderr);
    return -4;
  }
  ldc = (m + 1u) & ~1u;
  if ((C = malloc(ldc * n * sizeof(double complex))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -5;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refB[j * ldb + i] = B[j * ldb + i] = (double)rand() / (double)RAND_MAX + ((double)rand() / (double)RAND_MAX) * I;
  }

  ztrmm_ref(side, uplo, trans, diag, m, n, alpha, A, lda, refB, ldb);
  ztrmm2(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
//   ztrmm_(&s, &u, &t, &d, &m, &n, &alpha, A, &lda, B, &ldb);

  bool passed = true;
  double rdiff = 0.0, idiff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(creal(C[j * ldc + i]) - creal(refB[j * ldb + i]));
      if (d > rdiff)
        rdiff = d;

      double c = fabs(cimag(C[j * ldc + i]) - cimag(refB[j * ldb + i]));
      if (c > idiff)
        idiff = c;

      size_t flops;
      if (side == CBlasLeft)
        flops = 2 * i + 1;
      else
        flops = 2 * j + 1;
      if (diag == CBlasNonUnit)
        flops++;
      flops *= 3;

      if (d > (double)flops * 2.0 * DBL_EPSILON ||
          c > (double)flops * 2.0 * DBL_EPSILON)
        passed = false;
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
//     ztrmm_(&s, &u, &t, &d, &m, &n, &alpha, A, &lda, B, &ldb);
    ztrmm2(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = 6 * m * n;
  if (alpha != 0.0 + 0.0 * I) {
    flops += (side == CBlasLeft) ? 2 * m * n * (2 * m - 1) : 2 * m * n * (2 * n - 1);
    if (diag == CBlasNonUnit) flops += 18 * m * n;
  }

  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e + %.3ei\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, rdiff, idiff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refB);

  return (int)!passed;
}
