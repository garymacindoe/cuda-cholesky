static void LINALG_FUNCTION2(gold, lacgv)(TYPE(vector) * x) {
  for (size_t i = 0; i < x->n; i++)
    FUNCTION(vector, Set)(x, i, conj(FUNCTION(vector, Get)(x, i)));
}

static void TEST_FUNCTION(lacgv)() {
  TYPE(vector) x, y;
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&x, n * ioff + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&y, n * ioff + joff));

  for (size_t i = 0; i < y.n; i++)
    FUNCTION(vector, Set)(&y, i, G(LITERAL(0, 0), LITERAL(1, 1)));
  FUNCTION(vector, Copy)(&x, &y);

  TYPE(subvector) x0 = FUNCTION(vector, Subvector)(&x, joff, ioff, n);
  TYPE(subvector) y0 = FUNCTION(vector, Subvector)(&y, joff, ioff, n);

  LINALG_FUNCTION(lacgv)(&x0.v);
  LINALG_FUNCTION2(gold, lacgv)(&y0.v);

  for (size_t i = 0; i < x.n; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      LINALG_FUNCTION(lacgv)(&x0.v);
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, STRING(BLAS_PREC) "lacgv,%.3e,,%.3e\n", time, (double)n / time);
  }

  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&x));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&y));
}

static void TEST_FUNCTION2(lapack, lacgv)() {
  TYPE(vector) x, y;
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&x, n * ioff + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&y, n * ioff + joff));

  for (size_t i = 0; i < y.n; i++)
    FUNCTION(vector, Set)(&y, i, G(LITERAL(0, 0), LITERAL(1, 1)));
  FUNCTION(vector, Copy)(&x, &y);

  TYPE(subvector) x0 = FUNCTION(vector, Subvector)(&x, joff, ioff, n);
  TYPE(subvector) y0 = FUNCTION(vector, Subvector)(&y, joff, ioff, n);

  LINALG_FUNCTION2(lapack, lacgv)(&x0.v);
  LINALG_FUNCTION2(gold, lacgv)(&y0.v);

  for (size_t i = 0; i < x.n; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      LINALG_FUNCTION2(lapack, lacgv)(&x0.v);
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, "lapack" STRING(BLAS_PREC) "lacgv,%.3e,,%.3e\n", time, (double)n / time);
  }

  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&x));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&y));
}

static void TEST_FUNCTION2(cu, lacgv)() {
  TYPE(vector) x, y;
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&x, n * ioff + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&y, n * ioff + joff));

  TYPE(CUvector) dx;
  CU_ERROR_CHECK_VOID(FUNCTION(cuVector, Create)(&dx, n * ioff + joff));

  for (size_t i = 0; i < y.n; i++)
    FUNCTION(vector, Set)(&y, i, G(LITERAL(0, 0), LITERAL(1, 1)));
  CU_ERROR_CHECK_VOID(FUNCTION(cuVector, CopyHtoD)(&dx, &y));

  TYPE(subvector) y0 = FUNCTION(vector, Subvector)(&y, joff, ioff, n);
  TYPE(CUsubvector) dx0 = FUNCTION(cuVector, Subvector)(&dx, joff, ioff, n);

  CU_ERROR_CHECK_VOID(LINALG_FUNCTION2(cu, lacgv)(&dx0.v, NULL));
  LINALG_FUNCTION2(gold, lacgv)(&y0.v);

  CU_ERROR_CHECK_VOID(FUNCTION(cuVector, CopyDtoH)(&x, &dx));

  for (size_t i = 0; i < x.n; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));

  if (outfile != NULL) {
    CUevent start, stop;
    CU_ERROR_CHECK_VOID(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK_VOID(cuEventCreate( &stop, CU_EVENT_DEFAULT));

    CU_ERROR_CHECK_VOID(cuEventRecord(start, 0));
    for (size_t i = 0; i < iterations; i++)
      CU_ERROR_CHECK_VOID(LINALG_FUNCTION2(cu, lacgv)(&dx0.v, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, "cu" STRING(BLAS_PREC) "lacgv,%.3e,,%.3e\n", time, (float)n / time);
  }

  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&x));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&y));
  CU_ERROR_CHECK_VOID(FUNCTION(cuVector, Destroy)(&dx));
}
