/**
 * Generates a random symmetric, square, positive-definite matrix in single
 * complex precision with a specific condition number.
 *
 * @param n    the size of the matrix (must be greater than 1)
 * @param c    the condition number (must not be less than 1)
 * @param A    the matrix
 * @param lda  the leading dimension of the matrix
 * @return zero on success, nonzero on failure
 */
static int clatmc(size_t n, float c, float complex * A, size_t lda) {
  int info = 0;
  if (n < 2)
    info = -1;
  else if (c < 1.0f)
    info = -2;
  else if (lda < n)
    info = -4;
  if (info != 0)
    return info;

  float complex * u, * v, * w;

  if ((u = malloc(3 * n * sizeof(float complex))) == NULL)
    return 1;

  v = &u[n];
  w = &v[n];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = 0.0f + 0.0f * I;
  }

  A[0] = 1.0f + 0.0f * I;
  A[lda + 1] = c + 0.0f * I;
  for (size_t j = 2; j < n; j++)
    A[j * lda + j] = ((float) rand() / (float)RAND_MAX) * (c - 1.0f) + 1.0f + 0.0f * I;

  float t = 0.0f;
  float complex s = 0.0f + 0.0f * I;
  for (size_t j = 0; j < n; j++) {
    // u is a complex precision random vector
    u[j] = ((float)rand() / (float)RAND_MAX) + ((float)rand() / (float)RAND_MAX) * I;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += conjf(u[j]) * u[j];
    // s = t^2 u'v / 2
    s += conjf(u[j]) * v[j];
  }
  t = 2.0f / t;
  s = t * t * s / (2.0f + 0.0f * I);

  // w = tv - su
  for (size_t j = 0; j < n; j++)
    w[j] = t * v[j] - s * u[j];

  // A -= uw' + wu'
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] -= u[i] * conjf(w[j]) + w[i] * conjf(u[j]);
  }

  free(u);

  return 0;
}
