static void spotrf_ref(CBlasUplo uplo, size_t n, float * restrict A, size_t lda, long * restrict info) {
  *info = 0;
  if (lda < n)
    *info = -4;
  if (*info != 0) {
    XERBLA(-(*info));
    return;
  }

  if (n == 0) return;

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = 0; i <= j; i++) {
        float temp = A[j * lda + i];
        for (size_t k = 0; k < i; k++)
          temp -= A[j * lda + k] * A[i * lda + k];
        if (i == j) {
          if (temp <= 0.0f || isnan(temp)) {
            A[j * lda + j] = temp;
            *info = (long)j;
            return;
          }
          A[j * lda + j] = sqrtf(temp);
        }
        else
          A[j * lda + i] = temp / A[i * lda + i];
      }
    }
  }
  else {
    for (size_t j = 0; j < n; j++) {
      for (size_t i = j; i < n; i++) {
        float temp = A[j * lda + i];
        for (size_t k = 0; k < j; k++)
          temp -= A[k * lda + j] * A[k * lda + i];
        if (i == j) {
          if (temp <= 0.0f || isnan(temp)) {
            A[j * lda + j] = temp;
            *info = (long)j;
            return;
          }
          A[j * lda + j] = sqrtf(temp);
        }
        else
          A[j * lda + i] = temp / A[j * lda + j];
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
  float phi = 2.f * 3.1415926535f * u1;
  next = r * sinf(phi);
  hasNext = true;

  return r * cosf(phi);
}

