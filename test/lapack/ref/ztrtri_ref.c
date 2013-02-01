static void ztrtri_ref(CBlasUplo uplo, CBlasDiag diag, size_t n,
                       double complex * restrict A, size_t lda, long * restrict info) {
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
      double complex ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == (0.0 + 0.0 * I)) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = (1.0 + 0.0 * I) / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -(1.0 + 0.0 * I);

      for (size_t i = 0; i < j; i++) {
        double complex temp = A[j * lda + i];
        if (diag == CBlasNonUnit) temp *= A[i * lda + i];
        for (size_t k = i + 1; k < j; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    }
  }
  else {
    size_t j = n - 1;
    do {
      double complex ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == (0.0 + 0.0 * I)) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = (1.0 + 0.0 * I) / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -(1.0 + 0.0 * I);

      for (size_t i = n - 1; i > j; i--) {
        double complex temp = A[j * lda + i];
        if (diag == CBlasNonUnit) temp *= A[i * lda + i];
        for (size_t k = j + 1; k < i; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    } while (j-- > 0);
  }
}
