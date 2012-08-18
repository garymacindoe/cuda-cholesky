#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

#ifndef SYRK_ERROR
#define SYRK_ERROR
static inline size_t syrk_flops(size_t n, size_t k, long double alpha, long double beta) {
  const size_t mul = 1, add = 1;
  size_t flops = k * mul + (k - 1) * add;
  if (alpha != 0.l) flops += mul;
  if (beta != 1.l) flops += mul + add;
  return ((n * (n + 1)) / 2) * flops;
}

static inline long double syrk_error(size_t k, long double alpha, long double beta, long double epsilon) {
  const size_t mul = 1, add = 1;
  long double error = (long double)(k * mul + (k - 1) * add) * epsilon + (long double)(k * mul + add) * LDBL_EPSILON;
  if (alpha != 0.l) error += (long double)mul * (epsilon + LDBL_EPSILON);
  if (beta != 1.l) error += (long double)(mul + add) * (epsilon + LDBL_EPSILON);
  return error;
}

static inline size_t syrk_flops_complex(size_t n, size_t k, long double alpha, long double beta) {
  const size_t mul = 6, add = 2;
  size_t flops = k * mul + (k - 1) * add;
  if (alpha != 0.l) flops += mul;
  if (beta != 1.l) flops += mul + add;
  return ((n * (n + 1)) / 2) * flops;
}

static inline long double syrk_error_complex(size_t k, long double alpha, long double beta, long double epsilon) {
  const size_t mul = 3, add = 1;
  long double error = (long double)(k * mul + (k - 1) * add) * epsilon + (long double)(k * mul + add) * LDBL_EPSILON;
  if (alpha != 0.l) error += (long double)mul * (epsilon + LDBL_EPSILON);
  if (beta != 1.l) error += (long double)(mul + add) * (epsilon + LDBL_EPSILON);
  return error;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR syrk_error_complex(k, alpha, beta, EPSILON)
#define IMAG_ERROR syrk_error_complex(k, alpha, beta, EPSILON)
#define FLOPS syrk_flops_complex(n, k, alpha, beta)
#define EXTENDED long double complex
#else
#define ERROR syrk_error(k, alpha, beta, EPSILON)
#define FLOPS syrk_flops(n, k, alpha, beta)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, syrk)(CBlasUplo uplo, CBlasTranspose trans, SCALAR alpha, const TYPE(matrix) * A, SCALAR beta, TYPE(matrix) * C) {
  if (trans == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = 0; i <= j; i++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;

          for (size_t k = 0; k < A->n; k++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, k) * (EXTENDED)FUNCTION(matrix, Get)(A, j, k)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }

          FUNCTION(matrix, Set)(C, i, j, (SCALAR)((EXTENDED)alpha * temp + (EXTENDED)beta * (EXTENDED)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
    else {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = j; i < C->m; i++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;

          for (size_t k = 0; k < A->n; k++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, k) * (EXTENDED)FUNCTION(matrix, Get)(A, j, k)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }

          FUNCTION(matrix, Set)(C, i, j, (SCALAR)((EXTENDED)alpha * temp + (EXTENDED)beta * (EXTENDED)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = 0; i <= j; i++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;

          for (size_t k = 0; k < A->m; k++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }

          FUNCTION(matrix, Set)(C, i, j, (SCALAR)((EXTENDED)alpha * temp + (EXTENDED)beta * (EXTENDED)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
    else {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = j; i < C->m; i++) {
          EXTENDED temp = 0.l;
          EXTENDED carry = 0.l;

          for (size_t k = 0; k < A->m; k++) {
            EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(A, k, j)) - carry;
            EXTENDED total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }

          FUNCTION(matrix, Set)(C, i, j, (SCALAR)((EXTENDED)alpha * temp + (EXTENDED)beta * (EXTENDED)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
  }
}

#undef EXTENDED

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(syrk)() {
#define F LINALG_FUNCTION(syrk)
#define PREFIX ""
#include "testsyrk.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(blas, syrk)() {
#define F LINALG_FUNCTION2(blas, syrk)
#define PREFIX "blas"
#include "testsyrk.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, syrk)() {
#define F LINALG_FUNCTION2(cu, syrk)
#define PREFIX "cu"
#include "testsyrk.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CUBLAS_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cublas, syrk)() {
#define F LINALG_FUNCTION2(cublas, syrk)
#define PREFIX "cublas"
#define CUBLAS
#include "testsyrk.c"
#undef CUBLAS
#undef PREFIX
#undef F
}

#undef GPU
#undef ERROR_FUNCTION

#ifdef HAS_CULA
#define CULA

#define ERROR_FUNCTION(x) CULA_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cula, syrk)() {
#define F LINALG_FUNCTION2(cula, syrk)
#define PREFIX "cula"
#include "testsyrk.c"
#undef PREFIX
#undef F
}

#define GPU
static void TEST_FUNCTION2(culaDevice, syrk)() {
#define F LINALG_FUNCTION2(culaDevice, syrk)
#define PREFIX "culaDevice"
#include "testsyrk.c"
#undef PREFIX
#undef F
}
#undef GPU
#undef ERROR_FUNCTION

#undef CULA
#endif

#ifdef COMPLEX
#undef REAL_ERROR
#undef IMAG_ERROR
#else
#undef ERROR
#endif
#undef FLOPS
#undef __SELF_INCLUDE

#else

#ifdef CUBLAS
  cublasHandle_t handle;
  CUBLAS_ERROR_CHECK_VOID(cublasCreate(&handle));
#endif

  SCALAR alpha, beta;
  TYPE(matrix) A, C, D;
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&A, max(n + joff, k + koff), max(k + koff, n + joff)));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&C, n + joff, n + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&D, n + joff, n + joff));

#ifdef COMPLEX
  alpha = G(LITERAL(0, 0), LITERAL(1, 1));
  beta = G(LITERAL(0, 0), LITERAL(1, 1));
#else
  alpha = G(LITERAL(0), LITERAL(1));
  beta = G(LITERAL(0), LITERAL(1));
#endif

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&D, &C);

  CONST_TYPE(submatrix) A0 = FUNCTION(matrix, SubmatrixConst)(&A, joff, koff, n, k);
  TYPE(submatrix) D0 = FUNCTION(matrix, Submatrix)(&D, ioff, joff, m, n);

#ifdef GPU
  TYPE(CUmatrix) dA, dC;
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Create)(&dA, max(n + joff, k + koff), max(k + koff, n + joff)));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Create)(&dC, m + ioff, n + joff));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));

  CONST_TYPE(CUsubmatrix) dA0 = FUNCTION(cuMatrix, SubmatrixConst)(&dA, joff, koff, n, k);
  TYPE(CUsubmatrix) dC0 = FUNCTION(cuMatrix, Submatrix)(&dC, ioff, joff, m, n);

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  TYPE(submatrix) C0 = FUNCTION(matrix, Submatrix)(&C, ioff, joff, m, n);
  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, syrk)(CBlasUpper, CBlasNoTrans, alpha, &A0.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&C, i, j), FUNCTION(matrix, Get)(&D, i, j), ERROR);
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
#ifdef CUBLAS
      ERROR_FUNCTION(F(handle, CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkUN,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkUN,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&D, &C);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));
#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m, NULL));
#endif
#endif
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, syrk)(CBlasLower, CBlasNoTrans, alpha, &A0.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&C, i, j), FUNCTION(matrix, Get)(&D, i, j), ERROR);
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
#ifdef CUBLAS
      ERROR_FUNCTION(F(handle, CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m));
#else
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &dA0.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkLN,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkLN,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&D, &C);

  CONST_TYPE(submatrix) A1 = FUNCTION(matrix, SubmatrixConst)(&A, koff, joff, k, n);
#ifdef GPU
  CONST_TYPE(CUsubmatrix) dA1 = FUNCTION(cuMatrix, SubmatrixConst)(&dA, koff, joff, k, n);
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));
#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &A1.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, syrk)(CBlasUpper, CBlasTrans, alpha, &A1.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&C, i, j), FUNCTION(matrix, Get)(&D, i, j), ERROR);
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
#ifdef CUBLAS
      ERROR_FUNCTION(F(handle, CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
      ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkUT,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasTrans, alpha, &A1.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkUT,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&D, &C);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));
#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &A1.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, syrk)(CBlasLower, CBlasTrans, alpha, &A1.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&C, i, j), FUNCTION(matrix, Get)(&D, i, j), ERROR);
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
#ifdef CUBLAS
      ERROR_FUNCTION(F(handle, CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkLT,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasTrans, alpha, &A1.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "syrkLT,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&C));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&D));
#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dA));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dC));
#ifdef CUBLAS
  CUBLAS_ERROR_CHECK_VOID(cublasDestroy(handle));
#endif
#endif

#endif
