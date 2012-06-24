#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

// TODO: proper error/performance analysis for lauu2
#ifndef LAUU2_ERROR
#define LAUU2_ERROR
static inline size_t lauu2_flops(size_t n) {
  const size_t mul = 1, add = 1;
  return n * n * mul * add;
}

static inline long double lauu2_error(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
static inline size_t lauu2_flops_complex(size_t n) {
  const size_t mul = 1, add = 1;
  return n * n * mul * add;
}

static inline long double lauu2_error_real(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}

static inline long double lauu2_error_imag(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(i, uplo) lauu2_error_real(i, uplo, n, EPSILON)
#define IMAG_ERROR(i, uplo) lauu2_error_imag(i, uplo, n, EPSILON)
#define FLOPS lauu2_flops_complex(n)
#define EXTENDED_BASE long double
#define EXTENDED long double complex
#else
#define ERROR(i, uplo) lauu2_error(i, uplo, n, EPSILON)
#define FLOPS lauu2_flops(n)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, lauu2)(CBlasUplo uplo, TYPE(matrix) * A, long * info) {
  *info = 0;
  if (A->m != A->n) {
    *info = -2;
    return;
  }

  if (uplo == CBlasUpper) {
    for (size_t i = 0; i < A->m; i++) {
#ifdef COMPLEX
      EXTENDED_BASE aii = FUNCTION(matrix, Get)(A, i, i);
#else
      EXTENDED aii = FUNCTION(matrix, Get)(A, i, i);
#endif
      if (i < A->m - 1) {
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;
        for (size_t j = i; j < A->n; j++) {
#ifdef COMPLEX
          EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, i, j)) * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)) - carry;
#else
          EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, j) * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)) - carry;
#endif
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
#ifdef COMPLEX
        FUNCTION(matrix, Set)(A, i, i, (SCALAR)(aii * aii + creal(temp)));
#else
        FUNCTION(matrix, Set)(A, i, i, (SCALAR)temp);
#endif

        for (size_t k = 0; k < i; k++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;
          for (size_t j = i + 1; j < A->n; j++) {
#ifdef COMPLEX
            EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, i, j)) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
#else
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, j) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
#endif
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          FUNCTION(matrix, Set)(A, k, i, (SCALAR)(temp + aii * (EXTENDED)FUNCTION(matrix, Get)(A, k, i)));
        }
      }
      else {
        for (size_t j = 0; j <= i; j++)
#ifdef COMPLEX
          FUNCTION(matrix, Set)(A, j, i, (SCALAR)(aii * creal((EXTENDED)FUNCTION(matrix, Get)(A, j, i)) + aii * cimag((EXTENDED)FUNCTION(matrix, Get)(A, j, i)) * I));
#else
          FUNCTION(matrix, Set)(A, j, i, (SCALAR)(aii * (EXTENDED)FUNCTION(matrix, Get)(A, j, i)));
#endif
      }
    }
  }
  else {
    for (size_t i = 0; i < A->m; i++) {
#ifdef COMPLEX
      EXTENDED_BASE aii = FUNCTION(matrix, Get)(A, i, i);
#else
      EXTENDED aii = FUNCTION(matrix, Get)(A, i, i);
#endif
      if (i < A->m - 1) {
        EXTENDED temp = 0.l;
        EXTENDED carry = 0.l;
        for (size_t j = i; j < A->n; j++) {
#ifdef COMPLEX
          EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, j, i)) * (EXTENDED)FUNCTION(matrix, Get)(A, j, i)) - carry;
#else
          EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, i) * (EXTENDED)FUNCTION(matrix, Get)(A, j, i)) - carry;
#endif
          EXTENDED total = temp + element;
          carry = (total - temp) - element;
          temp = total;
        }
#ifdef COMPLEX
        FUNCTION(matrix, Set)(A, i, i, (SCALAR)(aii * aii + creal(temp)));
#else
        FUNCTION(matrix, Set)(A, i, i, (SCALAR)temp);
#endif

        for (size_t j = 0; j < i; j++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;
          for (size_t k = i + 1; k < A->m; k++) {
#ifdef COMPLEX
            EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, k, i)) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
#else
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
#endif
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          FUNCTION(matrix, Set)(A, i, j, (SCALAR)(temp + aii * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)));
        }
      }
      else {
        for (size_t j = 0; j <= i; j++)
#ifdef COMPLEX
          FUNCTION(matrix, Set)(A, i, j, (SCALAR)(aii * creal((EXTENDED)FUNCTION(matrix, Get)(A, i, j)) + aii * cimag((EXTENDED)FUNCTION(matrix, Get)(A, i, j)) * I));
#else
          FUNCTION(matrix, Set)(A, i, j, (SCALAR)(aii * (EXTENDED)FUNCTION(matrix, Get)(A, i, j)));
#endif
      }
    }
  }
}

#undef EXTENDED
#ifdef COMPLEX
#undef EXTENDED_BASE
#endif

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(lauu2)() {
#define F LINALG_FUNCTION(lauu2)
#define PREFIX ""
#include "testlauu2.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(lapack, lauu2)() {
#define F LINALG_FUNCTION2(lapack, lauu2)
#define PREFIX "lapack"
#include "testlauu2.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, lauu2)() {
#define F LINALG_FUNCTION2(cu, lauu2)
#define PREFIX "cu"
#include "testlauu2.c"
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

  ERROR_FUNCTION(F(CBlasUpper, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  TYPE(submatrix) A0 = FUNCTION(matrix, Submatrix)(&A, joff, joff, n, n);

  ERROR_FUNCTION(F(CBlasUpper, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, lauu2)(CBlasUpper, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasUpper));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasUpper));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasUpper));
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
      ERROR_FUNCTION(F(CBlasUpper, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "lauu2U,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "lauu2U,%.3e,,%.3e\n", time, (double)FLOPS / time);
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

  ERROR_FUNCTION(F(CBlasLower, &dA0.m, dInfo, NULL));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&A, &dA));
  CU_ERROR_CHECK_VOID(cuMemcpyDtoH(&infoA, dInfo, sizeof(long)));
#else
  ERROR_FUNCTION(F(CBlasLower, &A0.m, &infoA));
#endif
  LINALG_FUNCTION2(gold, lauu2)(CBlasLower, &B0.m, &infoB);

  CU_ASSERT_EQUAL(infoB, infoA);
  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&A, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, CBlasLower));
      CU_ASSERT_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&A, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, CBlasLower));
#else
      CU_ASSERT_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&A, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, CBlasLower));
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
      ERROR_FUNCTION(F(CBlasLower, &dA0.m, dInfo, NULL));
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "lauu2L,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "lauu2L,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&B));
#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dA));
  CU_ERROR_CHECK_VOID(cuMemFree(dInfo));
#endif

#endif
