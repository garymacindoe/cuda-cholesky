#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

// TODO: proper error/performance analysis for potrf
#ifndef POTRF_ERROR
#define POTRF_ERROR
static inline size_t potrf_flops(size_t n) {
  const size_t mul = 1, add = 1;
  return n * n * mul * add;
}

static inline long double potrf_error(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
static inline size_t potrf_flops_complex(size_t n) {
  const size_t mul = 1, add = 1;
  return n * n * mul * add;
}

static inline long double potrf_error_real(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}

static inline long double potrf_error_imag(size_t i, CBlasUplo uplo, size_t n, long double epsilon) {
  const size_t mul = 1, add = 1;
  if (uplo == CBlasLower)
    i = n - i;
  return (i * mul + (i - 1) * add) * epsilon;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(i, uplo) potrf_error_real(i, uplo, n, EPSILON)
#define IMAG_ERROR(i, uplo) potrf_error_imag(i, uplo, n, EPSILON)
#define FLOPS potrf_flops_complex(n)
#define EXTENDED_BASE long double
#define EXTENDED long double complex
#else
#define ERROR(i, uplo) potrf_error(i, uplo, n, EPSILON)
#define FLOPS potrf_flops(n)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, potrf)(CBlasUplo uplo, TYPE(matrix) * A, long * info) {
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
      EXTENDED carry = 0.l;
      for (size_t k = 0; k < i; k++) {
#ifdef COMPLEX
        EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, k, i)) * (EXTENDED)FUNCTION(matrix, Get)(A, k, i)) + carry;
#else
        EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(A, k, i)) + carry;
#endif
        EXTENDED total = aii - element;
        carry = (total - aii) + element;
        aii = total;
      }

      if (aii <= 0.l || isnan(aii)) {
        FUNCTION(matrix, Set)(A, i, i, (SCALAR)aii);
        *info = (long)i;
        break;
      }
      aii = sqrt(aii);
      FUNCTION(matrix, Set)(A, i, i, (SCALAR)aii);

      if (i < A->m - 1) {
        for (size_t j = i + 1; j < A->n; j++) {
          EXTENDED temp = FUNCTION(matrix, Get)(A, i, j);
          for (size_t k = 0; k < i; k++) {
#ifdef COMPLEX
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * conj((EXTENDED)FUNCTION(matrix, Get)(A, k, i))) + carry;
#else
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * (EXTENDED)FUNCTION(matrix, Get)(A, k, i)) + carry;
#endif
            EXTENDED total = temp - element;
            carry = (total - temp) + element;
            temp = total;
          }
          FUNCTION(matrix, Set)(A, i, j, (SCALAR)(temp / aii));
        }
      }
    }
  }
  else {
    for (size_t j = 0; j < A->n; j++) {
#ifdef COMPLEX
      EXTENDED_BASE ajj = FUNCTION(matrix, Get)(A, j, j);
#else
      EXTENDED ajj = FUNCTION(matrix, Get)(A, j, j);
#endif
      EXTENDED carry = 0.l;
      for (size_t k = 0; k < j; k++) {
#ifdef COMPLEX
        EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, j, k)) * (EXTENDED)FUNCTION(matrix, Get)(A, j, k)) + carry;
#else
        EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, k) * (EXTENDED)FUNCTION(matrix, Get)(A, j, k)) + carry;
#endif
        EXTENDED total = ajj - element;
        carry = (total - ajj) + element;
        ajj = total;
      }

      if (ajj <= 0.l || isnan(ajj)) {
        FUNCTION(matrix, Set)(A, j, j, (SCALAR)ajj);
        *info = (long)j;
        break;
      }
      ajj = sqrt(ajj);
      FUNCTION(matrix, Set)(A, j, j, (SCALAR)ajj);

      if (j < A->n - 1) {
        for (size_t i = j + 1; i < A->m; i++) {
          EXTENDED temp = FUNCTION(matrix, Get)(A, i, j);
          carry = 0.l;
          for (size_t k = 0; k < j; k++) {
#ifdef COMPLEX
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, k) * conj((EXTENDED)FUNCTION(matrix, Get)(A, i, k))) + carry;
#else
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, k) * (EXTENDED)FUNCTION(matrix, Get)(A, i, k)) + carry;
#endif
            EXTENDED total = ajj - element;
            carry = (total - ajj) + element;
            ajj = total;
          }
          FUNCTION(matrix, Set)(A, i, j, (SCALAR)(temp / ajj));
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

static void TEST_FUNCTION(potrf)() {
#define F LINALG_FUNCTION(potrf)
#define PREFIX ""
#include "testpotrf.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(lapack, potrf)() {
#define F LINALG_FUNCTION2(lapack, potrf)
#define PREFIX "lapack"
#include "testpotrf.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(cuMultiGPU, potrf)() {
#define F LINALG_FUNCTION2(cuMultiGPU, potrf)
#define PREFIX "cuMultiGPU"
#include "testpotrf.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, potrf)() {
#define F LINALG_FUNCTION2(cu, potrf)
#define PREFIX "cu"
#include "testpotrf.c"
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
  LINALG_FUNCTION2(gold, potrf)(CBlasUpper, &B0.m, &infoB);

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

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "potrfU,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "potrfU,%.3e,,%.3e\n", time, (double)FLOPS / time);
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
  LINALG_FUNCTION2(gold, potrf)(CBlasLower, &B0.m, &infoB);

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

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "potrfL,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, &A0.m, &infoA));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "potrfL,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&B));
#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dA));
  CU_ERROR_CHECK_VOID(cuMemFree(dInfo));
#endif

#endif
