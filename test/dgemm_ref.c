static void dgemm_ref(CBlasTranspose transA, CBlasTranspose transB, size_t m,
                      size_t n, size_t k, double alpha, const double * restrict A,
                      size_t lda, const double * restrict B, size_t ldb,
                      double beta, double * restrict C, size_t ldc) {

  if (m == 0 || n == 0 || ((k == 0 || alpha == 0.0) && beta == 1.0)) return;

  if (alpha == 0.0) {
    if (beta == 0.0) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = 0.0;
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

      double temp;
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

      if (alpha != 1.0)
        temp *= alpha;
      if (beta != 0.0)
        temp += beta * C[j * ldc + i];

      C[j * ldc + i] = temp;

    }
  }
}
