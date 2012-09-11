static void dtrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans,
                      CBlasDiag diag, size_t m, size_t n,
                      double alpha, const double * restrict A, size_t lda,
                      double * restrict B, size_t ldb, size_t * E) {

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
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = i + 1; k < m; k++) {
              temp -= A[k * lda + i] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = 0; k < i; k++) {
              temp -= A[k * lda + i] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = 0; k < i; k++) {
              temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = i + 1; k < m; k++) {
              temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
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
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = 0; k < j; k++) {
              temp -= A[j * lda + k] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        size_t j = n - 1;
        do {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = j + 1; k < n; k++) {
              temp -= A[j * lda + k] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
    }
    else {
      if (uplo == CBlasUpper) {
        size_t j = n - 1;
        do {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = j + 1; k < n; k++) {
              temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            for (size_t k = 0; k < j; k++) {
              temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            B[j * ldb + i] = temp;
          }
        }
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
