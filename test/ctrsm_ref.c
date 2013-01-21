static void ctrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans,
                      CBlasDiag diag, size_t m, size_t n,
                      float complex alpha, const float complex * restrict A, size_t lda,
                      float complex * restrict B, size_t ldb, size_t * restrict E) {

  if (m == 0 || n == 0) return;

  if (alpha == 0.0f + 0.0f * I) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i < m; i++)
        B[j * ldb + i] = 0.0f + 0.0f * I;
    }
    return;
  }

  if (side == CBlasLeft) {
    if (trans == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
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
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
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
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < i; k++) {
                temp -= conjf(A[i * lda + k]) * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
              if (diag == CBlasNonUnit) { temp /= conjf(A[i * lda + i]); E[j * ldb + i]++; }
            }
            else {
              for (size_t k = 0; k < i; k++) {
                temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
              if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
            }
            B[j * ldb + i] = temp;
          }
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          size_t i = m - 1;
          do {
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            if (trans == CBlasConjTrans) {
              for (size_t k = i + 1; k < m; k++) {
                temp -= conjf(A[i * lda + k]) * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
              if (diag == CBlasNonUnit) { temp /= conjf(A[i * lda + i]); E[j * ldb + i]++; }
            }
            else {
              for (size_t k = i + 1; k < m; k++) {
                temp -= A[i * lda + k] * B[j * ldb + k]; E[j * ldb + i] += E[j * ldb + k] + 2; }
              if (diag == CBlasNonUnit) { temp /= A[i * lda + i]; E[j * ldb + i]++; }
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
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
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
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
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
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            if (trans == CBlasConjTrans) {
              for (size_t k = j + 1; k < n; k++) {
                temp -= conjf(A[k * lda + j]) * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
              if (diag == CBlasNonUnit) { temp /= conjf(A[j * lda + j]); E[j * ldb + i]++; }
            }
            else {
              for (size_t k = j + 1; k < n; k++) {
                temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
              if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            }
            B[j * ldb + i] = temp;
          }
        } while (j-- > 0);
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < m; i++) {
            float complex temp = alpha * B[j * ldb + i]; E[j * ldb + i] = 1;
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < j; k++) {
                temp -= conjf(A[k * lda + j]) * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
              if (diag == CBlasNonUnit) { temp /= conjf(A[j * lda + j]); E[j * ldb + i]++; }
            }
            else {
              for (size_t k = 0; k < j; k++) {
                temp -= A[k * lda + j] * B[k * ldb + i]; E[j * ldb + i] += E[k * ldb + i] + 2; }
              if (diag == CBlasNonUnit) { temp /= A[j * lda + j]; E[j * ldb + i]++; }
            }
            B[j * ldb + i] = temp;
          }
        }
      }
    }
  }
}
