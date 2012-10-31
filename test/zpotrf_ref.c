static void zpotrf_ref(CBlasUplo uplo, size_t n, double complex * restrict A, size_t lda, long * restrict info) {
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
      for (size_t i = 0; i < j; i++) {
        double complex temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * conj(A[i * lda + k]);
        A[j * lda + i] = temp / A[i * lda + i];
      }

      double ajj = creal(A[j * lda + j]);
      for (size_t k = 0; k < j; k++)
        ajj -= A[j * lda + k] * conj(A[j * lda + k]);
      if (ajj <= 0.0 || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j + 1;
        return;
      }
      else
        A[j * lda + j] = sqrt(ajj);
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < j; k++) {
        double complex temp = conj(A[k * lda + j]);
        for (size_t i = j; i < n; i++)
          A[j * lda + i] -= temp * A[k * lda + i];
      }

      double ajj = creal(A[j * lda + j]);
      if (ajj <= 0.0 || isnan(ajj)) {
        A[j * lda + j] = ajj;
        *info = (long)j + 1;
        return;
      }
      ajj = sqrt(ajj);
      A[j * lda + j] = ajj;
      for (size_t i = j + 1; i < n; i++)
        A[j * lda + i] /= ajj;
    }
  }
}

static double complex gaussian() {
  double u0 = ((double)rand() + 1) / (double)RAND_MAX;
  double u1 = ((double)rand() + 1) / (double)RAND_MAX;
  double r = sqrt(-2 * log(u0));
  double phi = 2. * 3.1415926535897932384626433832795 * u1;
  double real = r * sin(phi);
  double imag = r * cos(phi);
  return real + imag * I;
}
