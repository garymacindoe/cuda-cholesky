#include "lapack.h"

static size_t min(size_t a, size_t b) { return (a < b) ? a : b; }

static const double complex zero = 0.0 + 0.0 * I;
static const double complex one = 1.0 + 0.0 * I;

static inline void ztrti2(CBlasUplo uplo, CBlasDiag diag,
                          size_t n,
                          const double complex * restrict A, size_t lda,
                          double complex * restrict B, size_t ldb,
                          long * restrict info) {
  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      register double complex bjj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        B[j * lda + j] = one / A[j * lda + j];
        bjj = -B[j * ldb + j];
      }
      else
        bjj = -one;

      for (size_t i = 0; i < j; i++) {
        B[j * ldb + i] = A[j * lda + i];
        if (A[j * lda + i] != zero) {
          register double complex temp = A[j * lda + i];
          for (size_t k = 0; k < i; k++)
            B[j * ldb + k] += temp * A[i * lda + k];
          if (diag == CBlasNonUnit) B[j * ldb + i] *= A[i * lda + i];
        }
      }
      for (size_t i = 0; i < j; i++)
        B[j * ldb + i] *= bjj;
    }
  }
  else {
    size_t j = n - 1;
    do {
      register double complex bjj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == zero) {
          *info = (long)j + 1;
          return;
        }
        B[j * ldb + j] = one / B[j * ldb + j];
        bjj = -B[j * ldb + j];
      }
      else
        bjj = -one;

      if (j < n - 1) {
        size_t i = n - 1;
        do {
        B[j * ldb + i] = A[j * lda + i];
          if (A[j * lda + i] != zero) {
            register double complex temp = A[j * lda + i];
            for (size_t k = i + 1; k < n; k++)
              B[j * ldb + k] += temp * A[i * lda + k];
            if (diag == CBlasNonUnit) B[j * ldb + i] *= A[i * lda + i];
          }
        } while (i-- > j + 1);
        for (size_t i = j + 1; i < n; i++)
          B[j * ldb + i] *= bjj;
      }
    } while (j-- > 0);
  }
}

void ztrtri2(CBlasUplo uplo, CBlasDiag diag,
             size_t n,
             const double complex * restrict A, size_t lda,
             double complex * restrict B, size_t ldb,
             long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (ldb < n)
    *info = -7;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  const size_t nb = 16;

  if (n < nb) {
    ztrti2(uplo, diag, n, A, lda, B, ldb, info);
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j += nb) {
      const size_t jb = min(nb, n - j);
      ztrmm2(CBlasLeft, CBlasUpper, CBlasNoTrans, diag,
             j, jb,
             one, B, ldb, &A[j * lda], lda,
             &B[j * ldb], ldb);
      ztrsm(CBlasRight, CBlasUpper, CBlasNoTrans, diag,
            j, jb,
            -one, &A[j * lda + j], lda,
            &B[j * ldb], ldb);
      ztrti2(CBlasUpper, diag,
             jb,
             &A[j * lda + j], lda,
             &B[j * ldb + j], ldb,
             info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    }
  }
  else {
    size_t j = (n + nb - 1) & ~(nb - 1);
    do {
      j -= nb;
      const size_t jb = min(nb, n - j);
      if (j + jb < n) {
        ztrmm2(CBlasLeft, CBlasLower, CBlasNoTrans, diag,
               n - j - jb, jb,
               one, &B[(j + jb) * ldb + j + jb], ldb, &A[j * lda + j + jb], lda,
               &B[j * ldb + j + jb], ldb);
        ztrsm(CBlasRight, CBlasLower, CBlasNoTrans, diag,
              n - j - jb, jb,
              -one, &A[j * lda + j], lda,
              &B[j * ldb + j + jb], ldb);
      }
      ztrti2(CBlasLower, diag,
             jb,
             &A[j * lda + j], lda,
             &B[j * ldb + j], ldb,
             info);
      if (*info != 0) {
        *info += (long)j;
        return;
      }
    } while (j > 0);
  }
}
