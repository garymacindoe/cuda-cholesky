#include "lapack.h"
#include "error.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>

static void dtrtri_ref(CBlasUplo, CBlasDiag, size_t, double * restrict, size_t, long * restrict);
static double gaussian();

int main(int argc, char * argv[]) {
  CBlasUplo uplo;
  CBlasDiag diag;
  size_t n;

  if (argc != 4) {
    fprintf(stderr, "Usage: %s <uplo> <diag> <n>\nwhere:\n"
                    "  uplo is 'u' or 'U' for CBlasUpper or 'l' or 'L' for CBlasLower\n"
                    "  diag is 'u' or 'U' for CBlasUnit or 'n' or 'N' for CBlasNonUnit\n"
                    "  n                  is the size of the matrix\n", argv[0]);
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

  double * A;
  size_t lda;
  long info;

  lda = (n + 1u) & ~1u;
  if ((A = malloc(lda *  n * sizeof(double))) == NULL) {
    fprintf(stderr, "Unable to allocate A\n");
    return -1;
  }

  double * C, * refA;
  size_t ldc, k = 5 * n;
  long rInfo;

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
      A[j * lda + i] = temp;
    }
  }
  free(C);

  dpotrf(uplo, n, A, lda, &info);
  if (info != 0) {
    fprintf(stderr, "Failed to compute Cholesky decomposition of A\n");
    return (int)info;
  }

  for (size_t j = 0; j < n; j++)
    memcpy(&refA[j * lda], &A[j * lda], n * sizeof(double));

  dtrtri_ref(uplo, diag, n, refA, lda, &rInfo);
  dtrtri(uplo, diag, n, A, lda, &info);

  bool passed = (info == rInfo);
  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      double d = fabs(A[j * lda + i] - refA[j * lda + i]);
      if (d > diff)
        diff = d;
    }
  }

  // Set A to identity so that repeated applications of the inverse
  // while benchmarking do not exit early due to singularity.
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
    dtrtri(uplo, diag, n, A, lda, &info);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  size_t flops = ((n * n * n) / 3) + ((2 * n) / 3);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time, ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(refA);

  return (int)!passed;
}

static void dtrtri_ref(CBlasUplo uplo, CBlasDiag diag, size_t n, double * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register double ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == 0.0) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = 1.0 / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -1.0;

      for (size_t i = 0; i < j; i++) {
        if (A[j * lda + i] != 0.0) {
          register double temp = A[j * lda + i];
          for (size_t k = 0; k < i; k++)
            A[j * lda + k] += temp * A[i * lda + k];
          if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
        }
      }
      for (size_t i = 0; i < j; i++)
        A[j * lda + i] *= ajj;
    }
  }
  else {
    size_t j = n - 1;
    do {
      register double ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == 0.0) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = 1.0 / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -1.0;

      if (j < n - 1) {
        size_t i = n - 1;
        do {
          if (A[j * lda + i] != 0.0) {
            register double temp = A[j * lda + i];
            if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
            for (size_t k = i + 1; k < n; k++)
              A[j * lda + k] += temp * A[i * lda + k];
          }
        } while (i-- > j + 1);
        for (size_t i = j + 1; i < n; i++)
          A[j * lda + i] *= ajj;
      }
    } while (j-- > 0);
  }
}

static double gaussian() {
  static bool hasNext = false;
  static double next;

  if (hasNext) {
    hasNext = false;
    return next;
  }

  double u0 = ((double)rand() + 1) / (double)RAND_MAX;
  double u1 = ((double)rand() + 1) / (double)RAND_MAX;
  double r = sqrt(-2 * log(u0));
  double phi = 2. * 3.1415926535897932384626433832795 * u1;
  next = r * sin(phi);
  hasNext = true;

  return r * cos(phi);
}
