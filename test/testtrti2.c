#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

// TODO: proper error/performance analysis for trti2
#ifndef TRTI2_ERROR
#define TRTI2_ERROR
static inline size_t trti2_flops(size_t n, CBlasDiag diag) {
  const size_t mul = 1, add = 1;
  (void)diag;
  return n * n * mul * add;
}

static inline long double trti2_error(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
static inline size_t trti2_flops_complex(size_t n, CBlasDiag diag) {
  const size_t mul = 1, add = 1;
  (void)diag;
  return n * n * mul * add;
}

static inline long double trti2_error_real(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}

static inline long double trti2_error_imag(size_t i, CBlasUplo uplo, CBlasDiag diag, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  (void)diag;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(i, uplo, diag) trti2_error_real(i, uplo, diag, n, EPSILON)
#define IMAG_ERROR(i, uplo, diag) trti2_error_imag(i, uplo, diag, n, EPSILON)
#define FLOPS(diag) trti2_flops_complex(n, diag)
#define EXTENDED_BASE long double
#define EXTENDED long double complex
#else
#define ERROR(i, uplo, diag) trti2_error(i, uplo, diag, n, EPSILON)
#define FLOPS(diag) trti2_flops(n, diag)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, trti2)(CBlasUplo uplo, CBlasDiag diag, TYPE(matrix) * A, long * info) {
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

static void TEST_FUNCTION(trti2)() {
#define F LINALG_FUNCTION(trti2)
#define PREFIX ""
#include "testtrti2.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(lapack, trti2)() {
#define F LINALG_FUNCTION2(lapack, trti2)
#define PREFIX "lapack"
#include "testtrti2.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, trti2)() {
#define F LINALG_FUNCTION2(cu, trti2)
#define PREFIX "cu"
#include "testtrti2.c"
#undef PREFIX
#undef F
}

#undef GPU
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

#ifdef GPU
  CUdeviceptr dInfo;
  CU_ERROR_CHECK_VOID(cuMemAlloc(&dInfo, sizeof(long)));

  TYPE(CUmatrix) dA;
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Create)(&dA, n + joff, n + joff));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));

  TYPE(CUsubmatrix) dA0 = FUNCTION(cuMatrix, Submatrix)(&dA, joff, joff, n, n);

  ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  TYPE(submatrix) A0 = FUNCTION(matrix, Submatrix)(&A, joff, joff, n, n);

  ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, trti2)(CBlasUpper, CBlasNonUnit, &B0.m, &infoB);

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
#ifdef GPU
    CUevent start, stop;
    CU_ERROR_CHECK_VOID(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK_VOID(cuEventCreate( &stop, CU_EVENT_DEFAULT));

    CU_ERROR_CHECK_VOID(cuEventRecord(start, 0));
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2UN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNonUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2UN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
#endif
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

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));

  ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, trti2)(CBlasUpper, CBlasUnit, &B0.m, &infoB);

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
#ifdef GPU
    CUevent start, stop;
    CU_ERROR_CHECK_VOID(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK_VOID(cuEventCreate( &stop, CU_EVENT_DEFAULT));

    CU_ERROR_CHECK_VOID(cuEventRecord(start, 0));
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2UU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2UU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
#endif
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

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));

  ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, trti2)(CBlasLower, CBlasNonUnit, &B0.m, &infoB);

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
#ifdef GPU
    CUevent start, stop;
    CU_ERROR_CHECK_VOID(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK_VOID(cuEventCreate( &stop, CU_EVENT_DEFAULT));

    CU_ERROR_CHECK_VOID(cuEventRecord(start, 0));
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2LN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNonUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2LN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasNonUnit) / time);
#endif
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

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));

  ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, trti2)(CBlasLower, CBlasUnit, &B0.m, &infoB);

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
#ifdef GPU
    CUevent start, stop;
    CU_ERROR_CHECK_VOID(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK_VOID(cuEventCreate( &stop, CU_EVENT_DEFAULT));

    CU_ERROR_CHECK_VOID(cuEventRecord(start, 0));
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2LU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasUnit, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trti2LU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasUnit) / time);
#endif
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&B));
#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dA));
  CU_ERROR_CHECK_VOID(cuMemFree(dInfo));
#endif

#endif
