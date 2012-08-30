static void sgemm_ref(CBlasTranspose transA, CBlasTranspose transB, size_t m,
                      size_t n, size_t k, float alpha, const float * restrict A,
                      size_t lda, const float * restrict B, size_t ldb,
                      float beta, float * restrict C, size_t ldc) {

  if (m == 0 || n == 0 || ((k == 0 || alpha == 0.0f) && beta == 1.0f)) return;

  if (alpha == 0.0f) {
    if (beta == 0.0f) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = 0.0f;
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

      float temp;
      if (transA == CBlasNoTrans) {
        if (transB == CBlasNoTrans) {
          temp = A[i] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[j * ldb + l];
        }
        else {
          temp = A[i] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[l * ldb + j];
        }
      }
      else {
        if (transB == CBlasNoTrans) {
          temp = A[i * lda] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[j * ldb + l];
        }
        else {
          temp = A[i * lda] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[l * ldb + j];
        }
      }

      if (alpha != 1.0f)
        temp *= alpha;
      if (beta != 0.0f)
        temp += beta * C[j * ldc + i];

      C[j * ldc + i] = temp;

    }
  }
}
