static void cgemm_ref(CBlasTranspose transA, CBlasTranspose transB,
                      size_t m, size_t n, size_t k,
                      float complex alpha, const float complex * restrict A, size_t lda,
                      const float complex * restrict B, size_t ldb,
                      float complex beta, float complex * restrict C, size_t ldc) {

  if (m == 0 || n == 0 || ((k == 0 || alpha == 0.0f + 0.0f * I) && beta == 1.0f + 0.0f * I))
    return;

  if (alpha == 0.0f + 0.0f * I) {
    if (beta == 0.0f + 0.0f * I) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = 0.0f + 0.0f * I;
      }
    }
    else {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = beta * C[j * ldc + i];
      }
    }
    return;
  }

  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < m; i++) {

      float complex temp;
      if (transA == CBlasNoTrans) {
        if (transB == CBlasNoTrans) {
          temp = A[i] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = A[i] * conjf(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conjf(B[l * ldb + j]);
        }
        else {
          temp = A[i] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[l * ldb + j];
        }
      }
      else if (transA == CBlasConjTrans) {
        if (transB == CBlasNoTrans) {
          temp = conjf(A[i * lda]) * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += conjf(A[i * lda + l]) * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = conjf(A[i * lda]) * conjf(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += conjf(A[i * lda + l]) * conjf(B[l * ldb + j]);
        }
        else {
          temp = conjf(A[i * lda]) * B[j];
          for (size_t l = 1; l < k; l++)
            temp += conjf(A[i * lda + l]) * B[l * ldb + j];
        }
      }
      else {
        if (transB == CBlasNoTrans) {
          temp = A[i * lda] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = A[i * lda] * conjf(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * conjf(B[l * ldb + j]);
        }
        else {
          temp = A[i * lda] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[l * ldb + j];
        }
      }

      if (alpha != 1.0f + 0.0f * I)
        temp *= alpha;
      if (beta != 0.0f + 0.0f * I)
        temp += beta * C[j * ldc + i];

      C[j * ldc + i] = temp;

    }
  }
}
