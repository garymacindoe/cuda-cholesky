#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <sys/time.h>
#include <float.h>

static void sgemm_ref(CBlasTranspose transA, CBlasTranspose transB, size_t m,
                      size_t n, size_t k, float alpha, const float * restrict A,
                      size_t lda, const float * restrict B, size_t ldb,
                      float beta, float * restrict C, size_t ldc) {

  if (m == 0 || n == 0 || ((k == 0 || alpha == 0.0f) && beta == 1.0f)) return;

  if (alpha == 0.0f) {
    if (beta == 0.0f) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = 0.0f;
      }
    }
    else {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = beta * C[j * ldc + i];
      }
    }
    return;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {

      float temp;
      if (transA == CBlasNoTrans) {
        if (transB == CBlasNoTrans) {
          temp = A[i] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[j * ldb + l];
        }
        else {
          temp = A[i] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[l * ldb + j];
        }
      }
      else {
        if (transB == CBlasNoTrans) {
          temp = A[i * lda] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[j * ldb + l];
        }
        else {
          temp = A[i * lda] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[l * ldb + j];
        }
      }

      if (alpha != 1.0f)
        temp *= alpha;
      if (beta != 0.0f)
        temp += beta * C[j * ldc + i];

      C[j * ldc + i] = temp;

    }
  }
}

int main(int argc, char * argv[]) {
  CBlasTranspose transA, transB;
  size_t m, n, k;

  if (argc != 6) {
    fprintf(stderr, "Usage: %s <transA> <transB> <m> <n> <k>\nwhere:\n  transA and transB  are 'n' or 'N' for CBlasNoTrans, 't' or 'T' for CBlasTrans or 'c' or 'C' for CBlasConjTrans\n  m, n and k         are the sizes of the matrices\n", argv[0]);
    return 1;
  }

  char t;
  if (sscanf(argv[1], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[1]);
    return 1;
  }
  switch (t) {
    case 'N': case 'n': transA = CBlasNoTrans; break;
    case 'T': case 't': transA = CBlasTrans; break;
    case 'C': case 'c': transA = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[2], "%c", &t) != 1) {
    fprintf(stderr, "Unable to read character from '%s'\n", argv[2]);
    return 2;
  }
  switch (t) {
    case 'N': case 'n': transB = CBlasNoTrans; break;
    case 'T': case 't': transB = CBlasTrans; break;
    case 'C': case 'c': transB = CBlasConjTrans; break;
    default: fprintf(stderr, "Unknown transpose '%c'\n", t); return 1;
  }

  if (sscanf(argv[3], "%zu", &m) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[3]);
    return 3;
  }

  if (sscanf(argv[4], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[4]);
    return 4;
  }

  if (sscanf(argv[5], "%zu", &k) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[5]);
    return 5;
  }

  srand(0);

  float alpha, beta, * A, * B, * C, * refC;
  size_t lda, ldb, ldc;

  alpha = (float)rand() / (float)RAND_MAX;
  beta = (float)rand() / (float)RAND_MAX;

  if (transA == CBlasNoTrans) {
    lda = (m + 15u) & ~15u;
    if ((A = malloc(lda * k * sizeof(float))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }
  }
  else {
    lda = (k + 15u) & ~15u;
    if ((A = malloc(lda * m * sizeof(float))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < k; i++)
        A[j * lda + i] = (float)rand() / (float)RAND_MAX;
    }
  }

  if (transB == CBlasNoTrans) {
    ldb = (k + 15u) & ~15u;
    if ((B = malloc(ldb * n * sizeof(float))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < k; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }
  }
  else {
    ldb = (n + 15u) & ~15u;
    if ((B = malloc(ldb * k * sizeof(float))) == NULL) {
      fputs("Unable to allocate B\n", stderr);
      return -2;
    }

    for (size_t j = 0; j < k; j++) {
      for (size_t i = 0; i < n; i++)
        B[j * ldb + i] = (float)rand() / (float)RAND_MAX;
    }
  }

  ldc = (m + 15u) & ~15u;
  if ((C = malloc(ldc * n * sizeof(float))) == NULL) {
    fputs("Unable to allocate C\n", stderr);
    return -3;
  }
  if ((refC = malloc(ldc * n * sizeof(float))) == NULL) {
    fputs("Unable to allocate refC\n", stderr);
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refC[j * ldc + i] = C[j * ldc + i] = (float)rand() / (float)RAND_MAX;
  }

  sgemm_ref(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, refC, ldc);
  sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  float diff = 0.0f;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      float d = fabsf(C[j * ldc + i] - refC[j * ldc + i]);
      if (d > diff)
        diff = d;
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = 2 * k - 1;
  if (alpha != 1.0f)
    flops += 1;
  if (beta != 0.0f)
    flops += 2;
  float error = (float)flops * 2.0f * FLT_EPSILON;
  flops *= m * n;

  bool passed = (diff <= error);
  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(C);
  free(refC);

  return (int)!passed;
}
