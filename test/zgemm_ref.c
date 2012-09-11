static void zgemm_ref(CBlasTranspose transA, CBlasTranspose transB, size_t m,
                      size_t n, size_t k, double complex alpha, const double complex * restrict A,
                      size_t lda, const double complex * restrict B, size_t ldb,
                      double complex beta, double complex * restrict C, size_t ldc) {

  if (m == 0 || n == 0 || ((k == 0 || alpha == 0.0 + 0.0 * I) && beta == 1.0 + 0.0 * I)) return;

  if (alpha == 0.0 + 0.0 * I) {
    if (beta == 0.0 + 0.0 * I) {
      for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < m; i++)
          C[j * ldc + i] = 0.0 + 0.0 * I;
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

      double complex temp;
      if (transA == CBlasNoTrans) {
        if (transB == CBlasNoTrans) {
          temp = A[i] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = A[i] * conj(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conj(B[l * ldb + j]);
        }
        else {
          temp = A[i] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * B[l * ldb + j];
        }
      }
      else if (transA == CBlasConjTrans) {
        if (transB == CBlasNoTrans) {
          temp = conj(A[i * lda]) * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += conj(A[i * lda + l]) * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = conj(A[i * lda]) * conj(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += conj(A[i * lda + l]) * conj(B[l * ldb + j]);
        }
        else {
          temp = conj(A[i * lda]) * B[j];
          for (size_t l = 1; l < k; l++)
            temp += conj(A[i * lda + l]) * B[l * ldb + j];
        }
      }
      else {
        if (transB == CBlasNoTrans) {
          temp = A[i * lda] * B[j * ldb];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[j * ldb + l];
        }
        else if (transB == CBlasConjTrans) {
          temp = A[i * lda] * conj(B[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * conj(B[l * ldb + j]);
        }
        else {
          temp = A[i * lda] * B[j];
          for (size_t l = 1; l < k; l++)
            temp += A[i * lda + l] * B[l * ldb + j];
        }
      }

      if (alpha != 1.0 + 0.0 * I)
        temp *= alpha;
      if (beta != 0.0 + 0.0 * I)
        temp += beta * C[j * ldc + i];

      C[j * ldc + i] = temp;

    }
  }
}
