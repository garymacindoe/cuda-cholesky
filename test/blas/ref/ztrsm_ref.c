static void ztrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag,
                      size_t m, size_t n,
                      double complex alpha, const double complex * restrict A, size_t lda,
                      double complex * restrict B, size_t ldb,
                      size_t * restrict E, size_t * restrict F) {

  if (m == 0 || n == 0)
    return;

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
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            for (size_t k = i + 1; k < m; k++) {
              temp -= A[k * lda + i] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            B[j * ldb + i] = temp;
          } while (i-- > 0);
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            for (size_t k = 0; k < i; k++) {
              temp -= A[k * lda + i] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
            if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            B[j * ldb + i] = temp;
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < i; k++) {
                temp -= conj(A[i * lda + k]) * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= conj(A[i * lda + i]); E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            else {
              for (size_t k = 0; k < i; k++) {
                temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            if (trans == CBlasConjTrans) {
              for (size_t k = i + 1; k < m; k++) {
                temp -= conj(A[i * lda + k]) * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= conj(A[i * lda + i]); E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            else {
              for (size_t k = i + 1; k < m; k++) {
                temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
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
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            for (size_t k = 0; k < j; k++) {
              temp -= A[j * lda + k] * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        size_t j = n - 1;
        do {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            for (size_t k = j + 1; k < n; k++) {
              temp -= A[j * lda + k] * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
            if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
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
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            if (trans == CBlasConjTrans) {
              for (size_t k = j + 1; k < n; k++) {
                temp -= conj(A[k * lda + j]) * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= conj(A[j * lda + j]); E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            else {
              for (size_t k = j + 1; k < n; k++) {
                temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            double complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = F[j * ldb + i] = 3;
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < j; k++) {
                temp -= conj(A[k * lda + j]) * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= conj(A[j * lda + j]); E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            else {
              for (size_t k = 0; k < j; k++) {
                temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[j * ldb + k] + 4; F[j * ldb + i] += F[j * ldb + k] + 4; }
              if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i] += 12; F[j * ldb + i] += 12; }
            }
            B[j * ldb + i] = temp;
          }
        }
      }
    }
  }
}
