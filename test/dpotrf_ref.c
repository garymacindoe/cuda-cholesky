static void dpotrf_ref(CBlasUplo uplo, size_t n,
                       double * restrict A, size_t lda, long * restrict info) {
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
      for (size_t i = 0; i <= j; i++) {
        double temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * A[i * lda + k];
        if (i == j) {
          if (temp <= 0.0 || isnan(temp)) {
            A[j * lda + j] = temp;
            *info = (long)j + 1;
            return;
          }
          A[j * lda + j] = sqrt(temp);
        }
        else
          A[j * lda + i] = temp / A[i * lda + i];
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        double temp = A[j * lda + i];
        for (size_t k = 0; k < j; k++)
          temp -= A[k * lda + j] * A[k * lda + i];
        if (i == j) {
          if (temp <= 0.0 || isnan(temp)) {
            A[j * lda + j] = temp;
            *info = (long)j + 1;
            return;
          }
          A[j * lda + j] = sqrt(temp);
        }
        else
          A[j * lda + i] = temp / A[j * lda + j];
      }
    }
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
