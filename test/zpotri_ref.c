static void zpotri_ref(CBlasUplo uplo, size_t n,
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
      if (A[j * lda + j] == (0.0 + 0.0 * I)) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = (1.0 + 0.0 * I) / A[j * lda + j];
      double complex ajj = -A[j * lda + j];

      for (size_t i = 0; i < j; i++) {
        double complex temp = A[j * lda + i] * A[i * lda + i];
        for (size_t k = i + 1; k < j; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    }

    for (size_t j = 0; j < n; j++) {
      double complex ajj = A[j * lda + j];
      for (size_t i = 0; i <= j; i++) {
        A[j * lda + i] *= ajj;
        for (size_t k = j + 1; k < n; k++)
          A[j * lda + i] += A[k * lda + i] * A[k * lda + j];
      }
    }
  }
  else {
    size_t j = n - 1;
    do {
      if (A[j * lda + j] == (0.0 + 0.0 * I)) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = (1.0 + 0.0 * I) / A[j * lda + j];
      double complex ajj = -A[j * lda + j];

      size_t i = n - 1;
      do {
        double complex temp = A[j * lda + i] * A[i * lda + i];
        for (size_t k = j + 1; k < i; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      } while (i-- > j);
    } while (j-- > 0);

    for (size_t i = 0; i < n; i++) {
      double complex aii = A[i * lda + i];
      for (size_t j = 0; j <= i; j++) {
        A[j * lda + i] *= aii;
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += A[i * lda + k] * A[j * lda + k];
      }
    }
  }
}

static double complex gaussian() {
  static bool hasNext = false;
  static double complex next;

  if (hasNext) {
    hasNext = false;
    return next;
  }

  double complex u0 = ((double complex)rand() + 1) / (double complex)RAND_MAX;
  double complex u1 = ((double complex)rand() + 1) / (double complex)RAND_MAX;
  double complex r = sqrtf(-2 * logf(u0));
  double complex phi = 2.0 * 3.1415926535f * u1;
  next = r * sinf(phi);
  hasNext = true;

  return r * cosf(phi);
}
