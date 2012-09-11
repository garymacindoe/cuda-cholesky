static void ctrsm_ref(CBlasSide side, CBlasUplo uplo, CBlasTranspose trans,
                      CBlasDiag diag, size_t m, size_t n,
                      float complex alpha, const float complex * restrict A, size_t lda,
                      float complex * restrict B, size_t ldb) {

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
            float complex temp = alpha * B[j * ldb + i];
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
            float complex temp = alpha * B[j * ldb + i];
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
            float complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < i; k++)
                temp -= conjf(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conjf(A[i * lda + i]);
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
            float complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = i + 1; k < m; k++)
                temp -= conjf(A[i * lda + k]) * B[j * ldb + k];
              if (diag == CBlasNonUnit) temp /= conjf(A[i * lda + i]);
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
            float complex temp = alpha * B[j * ldb + i];
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
            float complex temp = alpha * B[j * ldb + i];
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
            float complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = j + 1; k < n; k++)
                temp -= conjf(A[k * lda + j]) * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= conjf(A[j * lda + j]);
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
            float complex temp = alpha * B[j * ldb + i];
            if (trans == CBlasConjTrans) {
              for (size_t k = 0; k < j; k++)
                temp -= conjf(A[k * lda + j]) * B[k * ldb + i];
              if (diag == CBlasNonUnit) temp /= conjf(A[j * lda + j]);
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

static float complex gaussian() {
  float u0 = ((float)rand() + 1) / (float)RAND_MAX;
  float u1 = ((float)rand() + 1) / (float)RAND_MAX;
  float r = sqrtf(-2 * logf(u0));
  float phi = 2.f * 3.1415926535f * u1;
  float real = r * sinf(phi);
  float imag = r * cosf(phi);
  return real + imag * I;
}
