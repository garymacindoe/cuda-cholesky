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

static double complex gaussian() {
  double u0 = ((double)rand() + 1.0) / (double)RAND_MAX;
  double u1 = ((double)rand() + 1.0) / (double)RAND_MAX;
  double r = sqrt(-2.0 * log(u0));
  double phi = 2.0 * 3.1415926535897932384626433832795 * u1;
  double real = r * sin(phi);
  double imag = r * cos(phi);
  return real + imag * I;
}
