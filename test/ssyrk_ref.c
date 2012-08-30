static void ssyrk_ref(CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k,
                      float alpha, const float * restrict A, size_t lda,
                      float beta, float * restrict C, size_t ldc) {

  if (n == 0 || ((k == 0 || alpha == 0.0f) && beta == 1.0f)) return;

  if (alpha == 0.0f) {
    if (uplo == CBlasUpper) {
      if (beta == 0.0f) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = 0.0f;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = beta * C[j * ldc + i];
        }
      }
    }
    else {
      if (beta == 0.0f) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = 0.0f;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = beta * C[j * ldc + i];
        }
      }
    }
    return;
  }

  for (size_t j = 0; j < n; j++) {
    if (uplo == CBlasUpper) {
      for (size_t i = 0; i <= j; i++) {
        float temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * A[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * A[l * lda + j];
        }
        else {
          temp = A[i * lda] * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
        }

        if (alpha != 1.0f)
          temp *= alpha;
        if (beta != 0.0f)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;
      }
    }
    else {
      for (size_t i = j; i < n; i++) {
        float temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * A[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * A[l * lda + j];
        }
        else {
          temp = A[i * lda] * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * A[j * lda + l];
        }

        if (alpha != 1.0f)
          temp *= alpha;
        if (beta != 0.0f)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;
      }
    }
  }
}
