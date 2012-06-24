#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

// TODO: proper error/performance analysis for trtri
#ifndef TRTRI_ERROR
#define TRTRI_ERROR
static inline size_t trtri_flops(size_t n, CBlasDiag diag) {
  const size_t mul = 1, add = 1;
  (void)diag;
  return n * n * mul * add;
}

static inline long double trtri_error(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}

static inline size_t trtri_flops_complex(size_t n, CBlasDiag diag) {
  const size_t mul = 1, add = 1;
  (void)diag;
  return n * n * mul * add;
}

static inline long double trtri_error_real(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}

static inline long double trtri_error_imag(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(i, uplo, diag) trtri_error_real(i, uplo, diag, n, EPSILON)
#define IMAG_ERROR(i, uplo, diag) trtri_error_imag(i, uplo, diag, n, EPSILON)
#define FLOPS(diag) trtri_flops_complex(n, diag)
#define EXTENDED_BASE long double
#define EXTENDED long double complex
#else
#define ERROR(i, uplo, diag) trtri_error(i, uplo, diag, n, EPSILON)
#define FLOPS(diag) trtri_flops(n, diag)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, trtri)(CBlasUplo uplo, CBlasDiag diag, TYPE(matrix) * A, long * info) {
  *info = 0;
  if (A->m != A->n) {
    *info = -3;
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t j = 0; j < A->n; j++) {
      EXTENDED ajj;
      if (diag == CBlasNonUnit) {
        FUNCTION(matrix, Set)(A, j, j, (SCALAR)(ajj = 1.l / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
        ajj = -ajj;
      }
      else
        ajj = -1.l;

      for (size_t i = 0; i < j; i++) {
        EXTENDED temp = FUNCTION(matrix, Get)(A, i, j);
        EXTENDED carry = 0.l;
        if (diag == CBlasNonUnit) temp *= (EXTENDED)FUNCTION(matrix, Get)(A, i, i);
        for (size_t k = i + 1; k < j; k++) {
          EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * (EXTENDED)FUNCTION(matrix, Get)(A, i, k)) - carry;
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
        FUNCTION(matrix, Set)(A, i, j, (SCALAR)(temp * ajj));
      }
    }
  }
  else {
    for (size_t _j = 0; _j < A->n; _j++) {
      size_t j = A->n - _j - 1;

      EXTENDED ajj;
      if (diag == CBlasNonUnit) {
        FUNCTION(matrix, Set)(A, j, j, (SCALAR)(ajj = 1.l / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
        ajj = -ajj;
      }
      else
        ajj = -1.l;

      if (j < A->n - 1) {
        for (size_t i = A->m - 1; i > j; i--) {
          EXTENDED temp = FUNCTION(matrix, Get)(A, i, j);
          EXTENDED carry = 0.l;
          if (diag == CBlasNonUnit) temp *= (EXTENDED)FUNCTION(matrix, Get)(A, i, i);
          for (size_t k = j + 1; k < i; k++) {
          EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * (EXTENDED)FUNCTION(matrix, Get)(A, i, k)) - carry;
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
        FUNCTION(matrix, Set)(A, i, j, (SCALAR)(temp * ajj));
        }
      }
    }
  }
}

#undef EXTENDED
#ifdef COMPLEX
#undef EXTENDED_BASE
#endif

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(trtri)() {
#define F LINALG_FUNCTION(trtri)
#define PREFIX ""
#include "testtrtri.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(lapack, trtri)() {
#define F LINALG_FUNCTION2(lapack, trtri)
#define PREFIX "lapack"
#include "testtrtri.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(cuMultiGPU, trtri)() {
#define F LINALG_FUNCTION2(cuMultiGPU, trtri)
#define PREFIX "cuMultiGPU"
#include "testtrtri.c"
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
  long infoA, infoB;
  TYPE(matrix) A, B;
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&A, n + joff, n + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&B, n + joff, n + joff));

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&B, &A);

  TYPE(submatrix) B0 = FUNCTION(matrix, Submatrix)(&B, joff, joff, n, n);
  TYPE(submatrix) A0 = FUNCTION(matrix, Submatrix)(&A, joff, joff, n, n);

  ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &A0.m, &infoA));
  LINALG_FUNCTION2(gold, trtri)(CBlasUpper, CBlasNonUnit, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasUpper, CBlasNonUnit));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasUpper, CBlasNonUnit));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasUpper, CBlasNonUnit));
#endif
    }
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trtriUN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&B, &A);

  ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &A0.m, &infoA));
  LINALG_FUNCTION2(gold, trtri)(CBlasUpper, CBlasUnit, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasUpper, CBlasUnit));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasUpper, CBlasUnit));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasUpper, CBlasUnit));
#endif
    }
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trtriUU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&B, &A);

  ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &A0.m, &infoA));
  LINALG_FUNCTION2(gold, trtri)(CBlasLower, CBlasNonUnit, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasLower, CBlasNonUnit));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasLower, CBlasNonUnit));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasLower, CBlasNonUnit));
#endif
    }
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trtriLN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
  }

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&B, &A);

  ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &A0.m, &infoA));
  LINALG_FUNCTION2(gold, trtri)(CBlasLower, CBlasUnit, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasLower, CBlasUnit));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasLower, CBlasUnit));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasLower, CBlasUnit));
#endif
    }
  }

  if (outfile != NULL) {
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trtriLU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&B));

#endif
