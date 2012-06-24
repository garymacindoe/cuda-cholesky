#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

#ifndef HERK_ERROR
#define HERK_ERROR
static inline size_t herk_flops(size_t n, size_t k, long double alpha, long double beta) {
  const size_t mul = 6, add = 2;
  size_t flops = k * mul + (k - 1) * add;
  if (alpha != 0.l) flops += mul;
  if (beta != 1.l) flops += mul + add;
  return ((n * (n + 1)) / 2) * flops;
}

static inline long double herk_error(size_t k, long double alpha, long double beta, long double epsilon) {
  const size_t mul = 3, add = 1;
  long double error = (long double)(k * mul + (k - 1) * add) * epsilon + (long double)(k * mul + add) * LDBL_EPSILON;
  if (alpha != 0.l) error += (long double)mul * (epsilon + LDBL_EPSILON);
  if (beta != 1.l) error += (long double)(mul + add) * (epsilon + LDBL_EPSILON);
  return error;
}
#endif

#define REAL_ERROR herk_error(k, alpha, beta, EPSILON)
#define IMAG_ERROR herk_error(k, alpha, beta, EPSILON)
#define FLOPS herk_flops(n, k, alpha, beta)

static void LINALG_FUNCTION2(gold, herk)(CBlasUplo uplo, CBlasTranspose trans, SCALAR alpha, const TYPE(matrix) * A, SCALAR beta, TYPE(matrix) * C) {
  if (trans == CBlasNoTrans) {
    if (uplo == CBlasUpper) {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = 0; i < j; i++) {
          long double complex temp = 0.l;
          long double complex carry = 0.l;
          for (size_t l = 0; l < A->n; l++) {
            long double complex element = ((long double complex)FUNCTION(matrix, Get)(A, i, l) * conj((long double complex)FUNCTION(matrix, Get)(A, j, l))) - carry;
            long double complex total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (beta == 0.0l)
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp));
          else
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp + (long double complex)beta * (long double complex)FUNCTION(matrix, Get)(C, i, j)));
        }
        long double rtemp = 0.l;
        long double rcarry = 0.l;
        for (size_t l = 0; l < A->n; l++) {
          long double element = ((long double complex)FUNCTION(matrix, Get)(A, j, l) * conj((long double complex)FUNCTION(matrix, Get)(A, j, l))) - rcarry;
          long double total = rtemp + element;
          rcarry = (total - rtemp) - element;
          rtemp = total;
        }
        if (beta == 0.l)
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp));
        else
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp + (long double)beta * creal((long double complex)FUNCTION(matrix, Get)(C, j, j))));
      }
    }
    else {
      for (size_t j = 0; j < C->n; j++) {
        long double rtemp = 0.l;
        long double rcarry = 0.l;
        for (size_t l = 0; l < A->n; l++) {
          long double element = ((long double complex)FUNCTION(matrix, Get)(A, j, l) * conj((long double complex)FUNCTION(matrix, Get)(A, j, l))) - rcarry;
          long double total = rtemp + element;
          rcarry = (total - rtemp) - element;
          rtemp = total;
        }
        if (beta == 0.l)
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp));
        else
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp + (long double)beta * creal((long double complex)FUNCTION(matrix, Get)(C, j, j))));
        for (size_t i = j + 1; i < C->m; i++) {
          long double complex temp = 0.l;
          long double complex carry = 0.l;
          for (size_t l = 0; l < A->n; l++) {
            long double complex element = ((long double complex)FUNCTION(matrix, Get)(A, i, l) * conj((long double complex)FUNCTION(matrix, Get)(A, j, l))) - carry;
            long double complex total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (beta == 0.0l)
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp));
          else
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp + (long double complex)beta * (long double complex)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
  }
  else {
    if (uplo == CBlasUpper) {
      for (size_t j = 0; j < C->n; j++) {
        for (size_t i = 0; i < j; i++) {
          long double complex temp = 0.l;
          long double complex carry = 0.l;
          for (size_t l = 0; l < A->m; l++) {
            long double complex element = (conj((long double complex)FUNCTION(matrix, Get)(A, l, i)) * (long double complex)FUNCTION(matrix, Get)(A, l, j)) - carry;
            long double complex total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (beta == 0.l)
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp));
          else
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp + (long double complex)beta * (long double complex)FUNCTION(matrix, Get)(C, i, j)));
        }
        long double rtemp = 0.l;
        long double rcarry = 0.l;
        for (size_t l = 0; l < A->m; l++) {
          long double element = (conj((long double complex)FUNCTION(matrix, Get)(A, l, j)) * (long double complex)FUNCTION(matrix, Get)(A, l, j)) - rcarry;
          long double total = rtemp + element;
          rcarry = (total - rtemp) - element;
          rtemp = total;
        }
        if (beta == 0.l)
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp));
        else
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp + (long double)beta * creal((long double complex)FUNCTION(matrix, Get)(C, j, j))));
      }
    }
    else {
      for (size_t j = 0; j < C->n; j++) {
        long double rtemp = 0.l;
        long double rcarry = 0.l;
        for (size_t l = 0; l < A->m; l++) {
          long double element = (conj((long double complex)FUNCTION(matrix, Get)(A, l, j)) * (long double complex)FUNCTION(matrix, Get)(A, l, j)) - rcarry;
          long double total = rtemp + element;
          rcarry = (total - rtemp) - element;
          rtemp = total;
        }
        if (beta == 0.l)
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp));
        else
          FUNCTION(matrix, Set)(C, j, j, (SCALAR)((long double)alpha * rtemp + (long double)beta * creal((long double complex)FUNCTION(matrix, Get)(C, j, j))));
        for (size_t i = j + 1; i < C->m; i++) {
          long double complex temp = 0.l;
          long double complex carry = 0.l;
          for (size_t l = 0; l < A->m; l++) {
            long double complex element = (conj((long double complex)FUNCTION(matrix, Get)(A, l, i)) * (long double complex)FUNCTION(matrix, Get)(A, l, j)) - carry;
            long double complex total = temp + element;
            carry = (total - temp) - element;
            temp = total;
          }
          if (beta == 0.l)
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp));
          else
            FUNCTION(matrix, Set)(C, i, j, (SCALAR)((long double complex)alpha * temp + (long double complex)beta * (long double complex)FUNCTION(matrix, Get)(C, i, j)));
        }
      }
    }
  }
}

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(herk)() {
#define F LINALG_FUNCTION(herk)
#define PREFIX ""
#include "testherk.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(blas, herk)() {
#define F LINALG_FUNCTION2(blas, herk)
#define PREFIX "blas"
#include "testherk.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, herk)() {
#define F LINALG_FUNCTION2(cu, herk)
#define PREFIX "cu"
#include "testherk.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CUBLAS_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cublas, herk)() {
#define F LINALG_FUNCTION2(cublas, herk)
#define PREFIX "cublas"
#define CUBLAS
#include "testherk.c"
#undef CUBLAS
#undef PREFIX
#undef F
}

#undef GPU
#undef ERROR_FUNCTION

#ifdef HAS_CULA
#define CULA

#define ERROR_FUNCTION(x) CULA_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cula, herk)() {
#define F LINALG_FUNCTION2(cula, herk)
#define PREFIX "cula"
#include "testherk.c"
#undef PREFIX
#undef F
}

#define GPU
static void TEST_FUNCTION2(culaDevice, herk)() {
#define F LINALG_FUNCTION2(culaDevice, herk)
#define PREFIX "culaDevice"
#include "testherk.c"
#undef PREFIX
#undef F
}
#undef GPU
#undef ERROR_FUNCTION

#undef CULA
#endif

#undef FLOPS
#undef REAL_ERROR
#undef IMAG_ERROR
#undef __SELF_INCLUDE

#else

#ifdef CUBLAS
  cublasHandle_t handle;
  CUBLAS_ERROR_CHECK_VOID(cublasCreate(&handle));
#endif

  BASE_TYPE alpha, beta;
  TYPE(matrix) A, C, D;
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&A, max(n + joff, k + koff), max(k + koff, n + joff)));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&C, n + joff, n + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&D, n + joff, n + joff));

  alpha = G(BASE_LITERAL(0), BASE_LITERAL(1));
  beta = G(BASE_LITERAL(0), BASE_LITERAL(1));

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
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
  LINALG_FUNCTION2(gold, herk)(CBlasUpper, CBlasNoTrans, alpha, &A0.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
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

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkUN,%.3e,,%.3e\n", time, (float)FLOPS / time);

#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkUN,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
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
  LINALG_FUNCTION2(gold, herk)(CBlasLower, CBlasNoTrans, alpha, &A0.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
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

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkLN,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasNoTrans, alpha, &A0.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkLN,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&D, &C);

  CONST_TYPE(submatrix) A1 = FUNCTION(matrix, SubmatrixConst)(&A, koff, joff, k, n);
#ifdef GPU
  CONST_TYPE(CUsubmatrix) dA1 = FUNCTION(cuMatrix, SubmatrixConst)(&dA, koff, joff, k, n);
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));
#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &A1.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, herk)(CBlasUpper, CBlasConjTrans, alpha, &A1.m, beta, &D0.m);

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++) {
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&C, i, j)), creal(FUNCTION(matrix, Get)(&D, i, j)), REAL_ERROR);
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&C, i, j)), cimag(FUNCTION(matrix, Get)(&D, i, j)), IMAG_ERROR);
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
      ERROR_FUNCTION(F(handle, CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
      ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkUC,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasUpper, CBlasConjTrans, alpha, &A1.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkUC,%.3e,,%.3e\n", time, (double)FLOPS / time);
#endif
  }

  for (size_t j = 0; j < C.n; j++) {
    for (size_t i = 0; i < C.m; i++)
      FUNCTION(matrix, Set)(&C, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&D, &C);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dC, &C));
#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&C, &dC));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &A1.m, beta, &C0.m));
#endif
  LINALG_FUNCTION2(gold, herk)(CBlasLower, CBlasConjTrans, alpha, &A1.m, beta, &D0.m);

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
      ERROR_FUNCTION(F(handle, CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m));
#else
  ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &dA1.m, beta, &dC0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkLC,%.3e,,%.3e\n", time, (float)FLOPS / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLower, CBlasConjTrans, alpha, &A1.m, beta, &C0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "herkLC,%.3e,,%.3e\n", time, (double)FLOPS / time);
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
