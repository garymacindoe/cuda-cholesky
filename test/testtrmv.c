#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

#ifndef TRMV_ERROR
#define TRMV_ERROR
static inline size_t trmv_flops(CBlasDiag diag, size_t n) {
  const size_t mul = 1, add = 1;

  size_t flops = ((n * (n + 1)) / 2) * mul + ((n * (n - 1)) / 2) * add;
  if (diag == CBlasUnit)
    flops -= n * mul;
  return flops;
}

static inline long double trmv_error(size_t j, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;

  long double error = ((n - j - 1) * (mul + add)) * epsilon + ((n - j - 1) * mul + 2 * add) * LDBL_EPSILON;
  if (diag == CBlasNonUnit)
    error += mul * (epsilon + LDBL_EPSILON);
  return error;
}

static inline size_t trmv_flops_complex(CBlasDiag diag, size_t n) {
  const size_t mul = 6, add = 2;

  size_t flops = ((n * (n + 1)) / 2) * mul + ((n * (n - 1)) / 2) * add;
  if (diag == CBlasUnit)
    flops -= n * mul;
  return flops;
}

static inline long double trmv_error_complex(size_t j, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 3, add = 1;

  long double error = ((n - j - 1) * (mul + add)) * epsilon + ((n - j - 1) * mul + 2 * add) * LDBL_EPSILON;
  if (diag == CBlasNonUnit)
    error += mul * (epsilon + LDBL_EPSILON);
  return error;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(j, diag) trmv_error_complex(j, diag, n, EPSILON)
#define IMAG_ERROR(j, diag) trmv_error_complex(j, diag, n, EPSILON)
#define FLOPS(diag) trmv_flops_complex(diag, n)
#define EXTENDED long double complex
#else
#define ERROR(j, diag) trmv_error(j, diag, n, EPSILON)
#define FLOPS(diag) trmv_flops(diag, n)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, trmv)(CBlasUplo uplo, CBlasTranspose trans, CBlasDiag diag, const TYPE(matrix) * A, TYPE(vector) * x) {
  if (trans == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
      for (size_t i = 0; i < A->m; i++) {
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;

        for (size_t j = i + 1; j < A->n; j++) {
          EXTENDED element = ((EXTENDED)FUNCTION(vector, Get)(x, j) * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)) - carry;
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
        if (diag == CBlasNonUnit)
          FUNCTION(vector, Set)(x, i, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, i) * (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
        FUNCTION(vector, Set)(x, i, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, i) + temp));
      }
    }
    else {
      for (size_t k = 0; k < A->m; k++) {
        size_t i = A->m - k - 1;
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;

        for (size_t j = 0; j < i; j++) {
          EXTENDED element = ((EXTENDED)FUNCTION(vector, Get)(x, j) * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)) - carry;
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
        if (diag == CBlasNonUnit)
          FUNCTION(vector, Set)(x, i, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, i) * (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
        FUNCTION(vector, Set)(x, i, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, i) + temp));
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t k = 0; k < A->n; k++) {
        size_t j = A->n - k - 1;
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;

#ifdef COMPLEX
        if (trans == CBlasTrans) {
#endif
          for (size_t k = 0; k < j; k++) {
            size_t i = j - k - 1;
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, j) * (EXTENDED)FUNCTION(vector, Get)(x, i)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (diag == CBlasNonUnit)
            FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) * (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
          FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) + temp));
#ifdef COMPLEX
        }
        else {
          for (size_t k = 0; k < j; k++) {
            size_t i = j - k - 1;
            EXTENDED element = conj((EXTENDED)FUNCTION(matrix, Get)(A, i, j)) * (EXTENDED)FUNCTION(vector, Get)(x, i) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (diag == CBlasNonUnit)
            FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) * conj((EXTENDED)FUNCTION(matrix, Get)(A, j, j))));
          FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) + temp));
        }
#endif
      }
    }
    else {
      for (size_t j = 0; j < A->n; j++) {
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;
#ifdef COMPLEX
        if (trans == CBlasTrans) {
#endif
          for (size_t i = j + 1; i < A->m; i++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, j) * (EXTENDED)FUNCTION(vector, Get)(x, i)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }

          if (diag == CBlasNonUnit)
            FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) * (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
          FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) + temp));
#ifdef COMPLEX
        }
        else {
          for (size_t i = j + 1; i < A->m; i++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, j) * (EXTENDED)FUNCTION(vector, Get)(x, i)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (diag == CBlasNonUnit)
            FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) * conj((EXTENDED)FUNCTION(matrix, Get)(A, j, j))));
          FUNCTION(vector, Set)(x, j, (SCALAR)((EXTENDED)FUNCTION(vector, Get)(x, j) + temp));
        }
#endif
      }
    }
  }
}

#undef EXTENDED

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(trmv)() {
#define F LINALG_FUNCTION(trmv)
#define PREFIX ""
#include "testtrmv.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(blas, trmv)() {
#define F LINALG_FUNCTION2(blas, trmv)
#define PREFIX "blas"
#include "testtrmv.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#undef FLOPS
#ifdef COMPLEX
#undef REAL_ERROR
#undef IMAG_ERROR
#else
#undef ERROR
#endif
#undef __SELF_INCLUDE

#else
  TYPE(matrix) A;
  TYPE(vector) x, y;
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&A, n + joff, n + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&x, n * increment + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Create)(&y, n * increment + joff));

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  CONST_TYPE(submatrix) A0 = FUNCTION(matrix, SubmatrixConst)(&A, joff, joff, n, n);
  TYPE(subvector) x0 = FUNCTION(vector, Subvector)(&x, joff, increment, n);
  TYPE(subvector) y0 = FUNCTION(vector, Subvector)(&y, joff, increment, n);

  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasNoTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasNonUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasNonUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasNonUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasNoTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasNonUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasNonUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasNonUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasConjTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasUpper, CBlasConjTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvUCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasNoTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasNonUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasNonUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasNonUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasNoTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasNonUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasNonUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasNonUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, CBlasNonUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasConjTrans, CBlasNonUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasNonUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasNonUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasNonUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, CBlasNonUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t i = 0; i < x.n; i++)
#ifdef COMPLEX
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
    FUNCTION(vector, Set)(&x, i, G(LITERAL(0), LITERAL(1)));
#endif
  FUNCTION(vector, Copy)(&y, &x);

  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, CBlasUnit, &A0.m, &x0.v));
  LINALG_FUNCTION2(gold, trmv)(CBlasLower, CBlasConjTrans, CBlasUnit, &A0.m, &y0.v);

  for (size_t i = 0; i < joff; i++)
    CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, i), FUNCTION(vector, Get)(&y, i));
  for (size_t i = 0; i < n; i++) {
    size_t j = joff + i * increment;
#ifdef COMPLEX
    CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(vector, Get)(&x, j)), creal(FUNCTION(vector, Get)(&y, j)), REAL_ERROR(i, CBlasUnit));
    CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(vector, Get)(&x, j)), cimag(FUNCTION(vector, Get)(&y, j)), IMAG_ERROR(i, CBlasUnit));
#else
    CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(vector, Get)(&x, j), FUNCTION(vector, Get)(&y, j), ERROR(i, CBlasUnit));
#endif
    const size_t limit = min(j + increment, x.n);
    for (size_t k = j + 1; k < limit; k++)
      CU_ASSERT_EQUAL(FUNCTION(vector, Get)(&x, k), FUNCTION(vector, Get)(&y, k));
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, CBlasUnit, &A0.m, &x0.v));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trmvLCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&x));
  CU_ERROR_CHECK_VOID(FUNCTION(vector, Destroy)(&y));
#endif
