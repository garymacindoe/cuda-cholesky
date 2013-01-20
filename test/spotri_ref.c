static void spotri_ref(CBlasUplo uplo, size_t n, float * restrict A, size_t lda, long * restrict info) {
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
      if (A[j * lda + j] == 0.0f) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = 1.0f / A[j * lda + j];
      float ajj = -A[j * lda + j];

      for (size_t i = 0; i < j; i++) {
        float temp = A[j * lda + i] * A[i * lda + i];
        for (size_t k = i + 1; k < j; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      }
    }

    for (size_t j = 0; j < n; j++) {
      float ajj = A[j * lda + j];
      for (size_t i = 0; i <= j; i++) {
        A[j * lda + i] *= ajj;
        for (size_t k = j + 1; k < n; k++)
          A[j * lda + i] += A[k * lda + i] * A[k * lda + j];
      }
    }
  }
  else {
    size_t j = n - 1;
    do {
      if (A[j * lda + j] == 0.0f) {
        *info = (long)j + 1;
        return;
      }
      A[j * lda + j] = 1.0f / A[j * lda + j];
      float ajj = -A[j * lda + j];

      size_t i = n - 1;
      do {
        float temp = A[j * lda + i] * A[i * lda + i];
        for (size_t k = j + 1; k < i; k++)
          temp += A[k * lda + i] * A[j * lda + k];
        A[j * lda + i] = temp * ajj;
      } while (i-- > j);
    } while (j-- > 0);

    for (size_t i = 0; i < n; i++) {
      float aii = A[i * lda + i];
      for (size_t j = 0; j <= i; j++) {
        A[j * lda + i] *= aii;
        for (size_t k = i + 1; k < n; k++)
          A[j * lda + i] += A[i * lda + k] * A[j * lda + k];
      }
    }
  }
}

static float gaussian() {
  static bool hasNext = false;
  static float next;

  if (hasNext) {
    hasNext = false;
    return next;
  }

  float u0 = ((float)rand() + 1) / (float)RAND_MAX;
  float u1 = ((float)rand() + 1) / (float)RAND_MAX;
  float r = sqrtf(-2 * logf(u0));
  float phi = 2.0f * 3.1415926535f * u1;
  next = r * sinf(phi);
  hasNext = true;

  return r * cosf(phi);
}
