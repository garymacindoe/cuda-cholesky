/**
 * Generates a random symmetric, square, positive-definite matrix in single
 * precision with a specific condition number.
 *
 * @param n    the size of the matrix (must be greater than 1)
 * @param c    the condition number (must not be less than 1)
 * @param A    the matrix
 * @param lda  the leading dimension of the matrix
 * @return zero on success, non-zero on failure
 */
static int dlatmc(size_t n, double c, double * A, size_t lda) {
  int info = 0;
  if (n < 2)
    info = -1;
  else if (c < 1.0)
    info = -2;
  else if (lda < n)
    info = -4;
  if (info != 0)
    return info;

  double * u, * v, * w;
  size_t offset = (n + 1u) & ~1u;

  if ((u = malloc(3 * offset * sizeof(double))) == NULL)
    return 1;

  v = &u[offset];
  w = &v[offset];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = 0.0;
  }

  A[0] = 1.0;
  A[lda + 1] = c;
  for (size_t j = 2; j < n; j++)
    A[j * lda + j] = ((double) rand() / (double)RAND_MAX) * (c - 1.0) + 1.0;

  double t = 0.0, s = 0.0;
  for (size_t j = 0; j < n; j++) {
    // u is a random vector
    u[j] = (double)rand() / (double)RAND_MAX;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += u[j] * u[j];
    // s = t^2 u'v / 2
    s += u[j] * v[j];
  }
  t = 2.0 / t;
  s = t * t * s / 2.0;

  // w = tv - su
  for (size_t j = 0; j < n; j++)
    w[j] = t * v[j] - s * u[j];

  // A -= uw' + wu'
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] -= u[i] * w[j] + w[i] * u[j];
  }

  free(u);

  return 0;
}
