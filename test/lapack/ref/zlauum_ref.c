static void zlauum_ref(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      double complex ajj = conj(A[j * lda + j]);
      for (size_t i = 0; i <= j; i++) {
        A[j * lda + i] *= ajj;
        for (size_t k = j + 1; k < n; k++)
          A[j * lda + i] += A[k * lda + i] * conj(A[k * lda + j]);
      }
    }
  }
  else {
    for (size_t i = 0; i < n; i++) {
      double complex aii = conj(A[i * lda + i]);
      for (size_t j = 0; j <= i; j++) {
        A[j * lda + i] *= aii;
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += conj(A[i * lda + k]) * A[j * lda + k];
      }
    }
  }
}
