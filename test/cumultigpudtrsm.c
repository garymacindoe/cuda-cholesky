#include "blas.h"
#include "error.h"
#include <stdio.h>
#include <sys/time.h>
#include <float.h>

static void dtrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans,
                      CBlasDiag diag, size_t m, size_t n,
                      double alpha, const double * restrict A, size_t lda,
                      double * restrict B, size_t ldb) {

  if (m == 0 || n == 0) return;

  if (alpha == 0.0) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = 0.0;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = i + 1; k < m; k++)
              temp -= A[k * lda + i] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = 0; k < i; k++)
              temp -= A[k * lda + i] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = 0; k < i; k++)
              temp -= A[i * lda + k] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = i + 1; k < m; k++)
              temp -= A[i * lda + k] * B[j * ldb + k];
            if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
    }
  }
  else {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = j + 1; k < n; k++)
              temp -= A[j * lda + k] * B[k * ldb + i];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = 0; k < j; k++)
              temp -= A[j * lda + k] * B[k * ldb + i];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = 0; k < j; k++)
              temp -= A[k * lda + j] * B[k * ldb + i];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i];
            for (size_t k = j + 1; k < n; k++)
              temp -= A[k * lda + j] * B[k * ldb + i];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
    }
  }
}

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
    case 'R': case 'r': side = CBlasLeft; break;
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

  double alpha, * A, * B, * refB;
  size_t lda, ldb;

  CU_ERROR_CHECK(cuInit(0));

  int deviceCount;
  CU_ERROR_CHECK(cuDeviceGetCount(&deviceCount));

  CUcontext contexts[deviceCount];
  for (int i = 0; i < deviceCount; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));
    CU_ERROR_CHECK(cuCtxCreate(&contexts[i], CU_CTX_BLOCKING_SYNC, device));
  }

  alpha = (double)rand() / (double)RAND_MAX;

  if (side == CBlasLeft) {
    lda = (m + 1u) & ~1u;
    if ((A = malloc(lda * m * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < m; j++) {
      for (size_t i = 0; i < m; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }
  else {
    lda = (n + 1u) & ~1u;
    if ((A = malloc(lda * n * sizeof(double))) == NULL) {
      fputs("Unable to allocate A\n", stderr);
      return -1;
    }

    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < n; i++)
        A[j * lda + i] = (double)rand() / (double)RAND_MAX;
    }
  }

  ldb = (m + 1u) & ~1u;
  if ((B = malloc(ldb * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate B\n", stderr);
    return -3;
  }
  if ((refB = malloc(ldb * n * sizeof(double))) == NULL) {
    fputs("Unable to allocate refB\n", stderr);
    return -4;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++)
      refB[j * ldb + i] = B[j * ldb + i] = (double)rand() / (double)RAND_MAX;
  }

  dtrsm_ref(side, uplo, trans, diag, m, n, alpha, A, lda, refB, ldb);
  CU_ERROR_CHECK(cuMultiGPUDtrsm(contexts, deviceCount, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));

  bool passed = true;
  double diff = 0.0;
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {
      double d = fabs(B[j * ldb + i] - refB[j * ldb + i]);
      if (d > diff)
        diff = d;

      if (passed) {
        size_t k;
        if (side == CBlasLeft) {
          if (uplo == CBlasUpper)
            k = (trans == CBlasNoTrans) ? m - i - 1 : i;
          else
            k = (trans == CBlasNoTrans) ? i : m - i - 1;
        }
        else {
          if (uplo == CBlasUpper)
            k = (trans == CBlasNoTrans) ? j : n - j - 1;
          else
            k = (trans == CBlasNoTrans) ? n - j - 1 : j;
        }
        if (diag == CBlasNonUnit)
          k++;
        if (alpha != 0.0)
          k++;

        if (d > (double)k * DBL_EPSILON)
          passed = false;
      }
    }
  }

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -5;
  }
  for (size_t i = 0; i < 20; i++)
    CU_ERROR_CHECK(cuMultiGPUDtrsm(contexts, deviceCount, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
  if (gettimeofday(&stop, NULL) != 0) {
    fputs("gettimeofday failed\n", stderr);
    return -6;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;

  size_t flops = m * n;
  if (alpha != 0.0f) {
    flops += (side == CBlasLeft) ? m * m * n : m * n * n;
    if (diag == CBlasNonUnit) flops += m * n;
  }

  fprintf(stdout, "%.3es %.3gGFlops/s Error: %.3e\n%sED!\n", time,
          ((double)flops * 1.e-9) / time, diff, (passed) ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(refB);

  for (int i = 0; i < deviceCount; i++)
    CU_ERROR_CHECK(cuCtxDestroy(contexts[i]));

  return (int)!passed;
}
