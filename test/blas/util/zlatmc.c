/**
 * Generates a random symmetric, square, positive-definite matrix in double
 * complex precision with a specific condition number.
 *
 * @param n    the size of the matrix (must be greater than 1)
 * @param c    the condition number (must not be less than 1)
 * @param A    the matrix
 * @param lda  the leading dimension of the matrix
 * @return zero on success, nonzero on failure
 */
static int zlatmc(size_t n, double c, double complex * A, size_t lda) {
  int info = 0;
  if (n < 2)
    info = -1;
  else if (c < 1.0)
    info = -2;
  else if (lda < n)
    info = -4;
  if (info != 0)
    return info;

  double complex * u, * v, * w;

  if ((u = malloc(3 * n * sizeof(double complex))) == NULL)
    return 1;

  v = &u[n];
  w = &v[n];

  // Initialise A as a diagonal matrix whose diagonal consists of numbers from
  // [1,c] with 1 and c chosen at least once (here in the top left)
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] = 0.0 + 0.0 * I;
  }

  A[0] = 1.0 + 0.0 * I;
  A[lda + 1] = c + 0.0 * I;
  for (size_t j = 2; j < n; j++)
    A[j * lda + j] = ((double) rand() / (double)RAND_MAX) * (c - 1.0) + 1.0 + 0.0 * I;

  double t = 0.0;
  double complex s = 0.0 + 0.0 * I;
  for (size_t j = 0; j < n; j++) {
    // u is a complex precision random vector
    u[j] = ((double)rand() / (double)RAND_MAX) + ((double)rand() / (double)RAND_MAX) * I;
    // v = Au
    v[j] = A[j * lda + j] * u[j];
    // t = 2/u'u
    t += conj(u[j]) * u[j];
    // s = t^2 u'v / 2
    s += conj(u[j]) * v[j];
  }
  t = 2.0 / t;
  s = t * t * s / (2.0 + 0.0 * I);

  // w = tv - su
  for (size_t j = 0; j < n; j++)
    w[j] = t * v[j] - s * u[j];

  // A -= uw' + wu'
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++)
      A[j * lda + i] -= u[i] * conj(w[j]) + w[i] * conj(u[j]);
  }

  free(u);

  return 0;
}
