static void ztrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans,
                      CBlasDiag diag, size_t m, size_t n,
                      double complex alpha, const double complex * restrict A, size_t lda,
                      double complex * restrict B, size_t ldb) {

  if (m == 0 || n == 0) return;

  if (alpha == 0.0 + 0.0 * I) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = 0.0 + 0.0 * I;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double complex temp = alpha * B[j * ldb + i];
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
            double complex temp = alpha * B[j * ldb + i];
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
            double complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < i; k++)
                temp -= conj(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conj(A[i * lda + i]);
            }
            else {
              for (size_t k = 0; k < i; k++)
                temp -= A[i * lda + k] * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = i + 1; k < m; k++)
                temp -= conj(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conj(A[i * lda + i]);
            }
            else {
              for (size_t k = i + 1; k < m; k++)
                temp -= A[i * lda + k] * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= A[i * lda + i];
            }
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
            double complex temp = alpha * B[j * ldb + i];
            for (size_t k = 0; k < j; k++)
              temp -= A[j * lda + k] * B[k * ldb + i];
            if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i];
            for (size_t k = j + 1; k < n; k++)
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
            double complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = j + 1; k < n; k++)
                temp -= conj(A[k * lda + j]) * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= conj(A[j * lda + j]);
            }
            else {
              for (size_t k = j + 1; k < n; k++)
                temp -= A[k * lda + j] * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < j; k++)
                temp -= conj(A[k * lda + j]) * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= conj(A[j * lda + j]);
            }
            else {
              for (size_t k = 0; k < j; k++)
                temp -= A[k * lda + j] * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= A[j * lda + j];
            }
            B[j * ldb + i] = temp;
          }
        }
      }
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
