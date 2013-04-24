static void zherk_ref(CBlasUplo uplo, CBlasTranspose trans,
                      size_t n, size_t k,
                      double alpha, const double complex * restrict A, size_t lda,
                      double beta, double complex * restrict C, size_t ldc) {

  if (n == 0 || ((k == 0 || alpha == 0.0) && beta == 1.0))
    return;

  if (alpha == 0.0) {
    if (uplo == CBlasUpper) {
      if (beta == 0.0) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i <= j; i++)
            C[j * ldc + i] = 0.0;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = 0; i < j; i++)
            C[j * ldc + i] = beta * C[j * ldc + i];
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
        }
      }
    }
    else {
      if (beta == 0.0) {
        for (size_t j = 0; j < n; j++) {
          for (size_t i = j; i < n; i++)
            C[j * ldc + i] = 0.0;
        }
      }
      else {
        for (size_t j = 0; j < n; j++) {
          C[j * ldc + j] = beta * creal(C[j * ldc + j]);
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
        double complex temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * conj(A[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conj(A[l * lda + j]);
        }
        else {
          temp = conj(A[i * lda]) * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += conj(A[i * lda + l]) * A[j * lda + l];
        }

        if (alpha != 1.0)
          temp *= alpha;
        if (beta != 0.0)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;
      }

      double rtemp;

      if (trans == CBlasNoTrans) {
        rtemp = A[j] * conj(A[j]);
        for (size_t l = 1; l < k; l++)
          rtemp += A[l * lda + j] * conj(A[l * lda + j]);
      }
      else {
        rtemp = conj(A[j * lda]) * A[j * lda];
        for (size_t l = 1; l < k; l++)
          rtemp += conj(A[j * lda + l]) * A[j * lda + l];
      }

      if (alpha != 1.0)
        rtemp *= alpha;
      if (beta != 0.0)
        rtemp += beta * C[j * ldc + j];

      C[j * ldc + j] = rtemp;
    }
    else {
      double rtemp;

      if (trans == CBlasNoTrans) {
        rtemp = A[j] * conj(A[j]);
        for (size_t l = 1; l < k; l++)
          rtemp += A[l * lda + j] * conj(A[l * lda + j]);
      }
      else {
        rtemp = conj(A[j * lda]) * A[j * lda];
        for (size_t l = 1; l < k; l++)
          rtemp += conj(A[j * lda + l]) * A[j * lda + l];
      }

      if (alpha != 1.0)
        rtemp *= alpha;
      if (beta != 0.0)
        rtemp += beta * C[j * ldc + j];

      C[j * ldc + j] = rtemp;

      for (size_t i = j + 1; i < n; i++) {
        double complex temp;

        if (trans == CBlasNoTrans) {
          temp = A[i] * conj(A[j]);
          for (size_t l = 1; l < k; l++)
            temp += A[l * lda + i] * conj(A[l * lda + j]);
        }
        else {
          temp = conj(A[i * lda]) * A[j * lda];
          for (size_t l = 1; l < k; l++)
            temp += conj(A[i * lda + l]) * A[j * lda + l];
        }

        if (alpha != 1.0)
          temp *= alpha;
        if (beta != 0.0)
          temp += beta * C[j * ldc + i];

        C[j * ldc + i] = temp;
      }
    }
  }
}
