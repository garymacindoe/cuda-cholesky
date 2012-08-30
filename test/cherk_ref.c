static void cherk_ref(CBlasUplo uplo, CBlasTranspose trans, size_t n, size_t k,
                      float alpha, const float complex * restrict A, size_t lda,
                      float beta, float complex * restrict C, size_t ldc) {

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
          for (size_t i = 0; i < j; i++)
            C[j * ldc + i] = beta * C[j * ldc + i];
          C[j * ldc + j] = beta * crealf(C[j * ldc + j]);
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
          C[j * ldc + j] = beta * crealf(C[j * ldc + j]);
          for (size_t i = j + 1; i < n; i++)
            C[j * ldc + i] = beta * C[j * ldc + i];
        }
      }
    }
    return;
  }

  for (size_t j = 0; j < n; j++) {
    if (uplo == CBlasUpper) {
      for (size_t i = 0; i < j; i++) {
        float complex temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * conjf(A[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conjf(A[l * lda + j]);
        }
        else {
          temp = conjf(A[i * lda]) * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += conjf(A[i * lda + l]) * A[j * lda + l];
        }

        if (alpha != 1.0f)
          temp *= alpha;
        if (beta != 0.0f)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;
      }

      float rtemp;

      if (trans == CBlasNoTrans) {
        rtemp = A[j] * conjf(A[j]);
        for (size_t l = 1; l < k; l++)
          rtemp += A[l * lda + j] * conjf(A[l * lda + j]);
      }
      else {
        rtemp = conjf(A[j * lda]) * A[j * lda];
        for (size_t l = 1; l < k; l++)
          rtemp += conjf(A[j * lda + l]) * A[j * lda + l];
      }

      if (alpha != 1.0f)
        rtemp *= alpha;
      if (beta != 0.0f)
        rtemp += beta * C[j * ldc + j];

      C[j * ldc + j] = rtemp;
    }
    else {
      float rtemp;

      if (trans == CBlasNoTrans) {
        rtemp = A[j] * conjf(A[j]);
        for (size_t l = 1; l < k; l++)
          rtemp += A[l * lda + j] * conjf(A[l * lda + j]);
      }
      else {
        rtemp = conjf(A[j * lda]) * A[j * lda];
        for (size_t l = 1; l < k; l++)
          rtemp += conjf(A[j * lda + l]) * A[j * lda + l];
      }

      if (alpha != 1.0f)
        rtemp *= alpha;
      if (beta != 0.0f)
        rtemp += beta * C[j * ldc + j];

      C[j * ldc + j] = rtemp;

      for (size_t i = j + 1; i < n; i++) {
        float complex temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * conjf(A[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conjf(A[l * lda + j]);
        }
        else {
          temp = conjf(A[i * lda]) * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += conjf(A[i * lda + l]) * A[j * lda + l];
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
