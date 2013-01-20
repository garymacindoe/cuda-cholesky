static void strtri_ref(CBlasUplo uplo, CBlasDiag diag, size_t n, float * restrict A, size_t lda, long * restrict info) {
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
      register float ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == 0.0f) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = 1.0f / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -1.0f;

      for (size_t i = 0; i < j; i++) {
        if (A[j * lda + i] != 0.0f) {
          register float temp = A[j * lda + i];
          for (size_t k = 0; k < i; k++)
            A[j * lda + k] += temp * A[i * lda + k];
          if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
        }
      }
      for (size_t i = 0; i < j; i++)
        A[j * lda + i] *= ajj;
    }
  }
  else {
    size_t j = n - 1;
    do {
      register float ajj;
      if (diag == CBlasNonUnit) {
        if (A[j * lda + j] == 0.0f) {
          *info = (long)j + 1;
          return;
        }
        A[j * lda + j] = 1.0f / A[j * lda + j];
        ajj = -A[j * lda + j];
      }
      else
        ajj = -1.0f;

      if (j < n - 1) {
        size_t i = n - 1;
        do {
          if (A[j * lda + i] != 0.0f) {
            register float temp = A[j * lda + i];
            if (diag == CBlasNonUnit) A[j * lda + i] *= A[i * lda + i];
            for (size_t k = i + 1; k < n; k++)
              A[j * lda + k] += temp * A[i * lda + k];
          }
        } while (i-- > j + 1);
        for (size_t i = j + 1; i < n; i++)
          A[j * lda + i] *= ajj;
      }
    } while (j-- > 0);
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
