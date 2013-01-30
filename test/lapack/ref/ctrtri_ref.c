static void ctrtri_ref(CBlasUplo uplo, CBlasDiag diag, size_t n,
                       float complex * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -5;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0)
    return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      float complex ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == (0.0f + 0.0f * I)) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = (1.0f + 0.0f * I) / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -(1.0f + 0.0f * I);

      for (size_t i = 0; i < j; i++) {
        float complex temp = A[j * lda + i];
        if (diag == CBlasNonUnit) temp *= A[i * lda + i];
        for (size_t k = i + 1; k < j; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    }
  }
  else {
    size_t j = n - 1;
    do {
      float complex ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == (0.0f + 0.0f * I)) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = (1.0f + 0.0f * I) / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -(1.0f + 0.0f * I);

      for (size_t i = n - 1; i > j; i--) {
        float complex temp = A[j * lda + i];
        if (diag == CBlasNonUnit) temp *= A[i * lda + i];
        for (size_t k = j + 1; k < i; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    } while (j-- > 0);
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
