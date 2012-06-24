#ifndef __SELF_INCLUDE
#define __SELF_INCLUDE

#ifndef TRSM_ERROR
#define TRSM_ERROR
static inline size_t trsm_flops(CBlasSide side, CBlasDiag diag, size_t m, size_t n, long double alpha) {
  const size_t div = 1, mul = 1, add = 1;

  const size_t inner = (side == CBlasLeft) ? m : n;
  const size_t outer = (side == CBlasRight) ? n : m;

  size_t flops = ((inner * (inner + 1)) / 2) * mul + ((inner * (inner - 1)) / 2) * add;
  if (diag == CBlasNonUnit)
    flops += inner * div;
  if (alpha != 0.l)
    flops += inner * mul;
  return outer * flops;
}

static inline long double trsm_error(size_t i, size_t j, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, long double alpha, long double epsilon) {
  const size_t div = 1, mul = 1, add = 1;

  size_t k;
  if (side == CBlasLeft) {
    if (uplo == CBlasUpper)
      k = (transA == CBlasNoTrans) ? m - i - 1 : i;
    else
      k = (transA == CBlasNoTrans) ? i : m - i - 1;
  }
  else {
    if (uplo == CBlasUpper)
      k = (transA == CBlasNoTrans) ? j : n - j - 1;
    else
      k = (transA == CBlasNoTrans) ? n - j - 1 : j;
  }

  long double error = (k * mul + (k - 1) * add) * epsilon + (k * mul + add) * LDBL_EPSILON;
  if (diag == CBlasNonUnit)
    error += div * (epsilon + LDBL_EPSILON);
  if (alpha != 0.l)
    error += mul * (epsilon + LDBL_EPSILON);
  return error;
}

static inline size_t trsm_flops_complex(CBlasSide side, CBlasDiag diag, size_t m, size_t n, long double alpha) {
  const size_t div = 20, mul = 6, add = 2;

  const size_t inner = (side == CBlasLeft) ? m : n;
  const size_t outer = (side == CBlasRight) ? n : m;

  size_t flops = ((inner * (inner + 1)) / 2) * mul + ((inner * (inner - 1)) / 2) * add;
  if (diag == CBlasNonUnit)
    flops += inner * div;
  if (alpha != 0.l)
    flops += inner * mul;
  return outer * flops;
}

static inline long double trsm_error_complex(size_t i, size_t j, CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, size_t m, size_t n, long double alpha, long double epsilon) {
  const size_t div = 16, mul = 3, add = 1;

  size_t k;
  if (side == CBlasLeft) {
    if (uplo == CBlasUpper)
      k = (transA == CBlasNoTrans) ? m - i - 1 : i;
    else
      k = (transA == CBlasNoTrans) ? i : m - i - 1;
  }
  else {
    if (uplo == CBlasUpper)
      k = (transA == CBlasNoTrans) ? j : n - j - 1;
    else
      k = (transA == CBlasNoTrans) ? n - j - 1 : j;
  }

  long double error = (k * mul + (k - 1) * add) * epsilon + (k * mul + add) * LDBL_EPSILON;
  if (diag == CBlasNonUnit)
    error += div * (epsilon + LDBL_EPSILON);
  if (alpha != 0.l)
    error += mul * (epsilon + LDBL_EPSILON);
  return error;
}
#endif

#ifdef COMPLEX
#define REAL_ERROR(i, j, side, uplo, transA, diag) trsm_error_complex(i, j, side, uplo, transA, diag, m, n, alpha, EPSILON)
#define IMAG_ERROR(i, j, side, uplo, transA, diag) trsm_error_complex(i, j, side, uplo, transA, diag, m, n, alpha, EPSILON)
#define FLOPS(side, diag) trsm_flops_complex(side, diag, m, n, alpha)
#define EXTENDED long double complex
#else
#define ERROR(i, j, side, uplo, transA, diag) trsm_error(i, j, side, uplo, transA, diag, m, n, alpha, EPSILON)
#define FLOPS(side, diag) trsm_flops(side, diag, m, n, alpha)
#define EXTENDED long double
#endif

static void LINALG_FUNCTION2(gold, trsm)(CBlasSide side, CBlasUplo uplo, CBlasTranspose transA, CBlasDiag diag, SCALAR alpha, const TYPE(matrix) * A, TYPE(matrix) * B) {
  if (side == CBlasLeft) {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t _i = 0; _i < B->m; _i++) {
            size_t i = B->m - _i - 1;
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
            for (size_t k = i + 1; k < B->m; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, k) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
          }
        }
      }
      else {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t i = 0; i < B->m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
            for (size_t k = 0; k < i; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, i, k) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t i = 0; i < B->m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
#ifdef COMPLEX
            if (transA == CBlasTrans) {
#endif
            for (size_t k = 0; k < i; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
#ifdef COMPLEX
            }
            else {
              for (size_t k = 0; k < i; k++) {
                EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, k, i)) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
                EXTENDED total = temp + element;
                carry = (total - temp) - element;
                temp = total;
              }
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
              if (diag == CBlasNonUnit)
                FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / conj((EXTENDED)FUNCTION(matrix, Get)(A, i, i))));
            }
#endif
          }
        }
      }
      else {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t _i = 0; _i < B->m; _i++) {
            size_t i = B->m - _i - 1;
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
#ifdef COMPLEX
            if (transA == CBlasTrans) {
#endif
            for (size_t k = i + 1; k < B->m; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, i) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, i, i)));
#ifdef COMPLEX
            }
            else {
              for (size_t k = i + 1; k < B->m; k++) {
                EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, k, i)) * (EXTENDED)FUNCTION(matrix, Get)(B, k, j)) - carry;
                EXTENDED total = temp + element;
                carry = (total - temp) - element;
                temp = total;
              }
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
              if (diag == CBlasNonUnit)
                FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / conj((EXTENDED)FUNCTION(matrix, Get)(A, i, i))));
            }
#endif
          }
        }
      }
    }
  }
  else {
    if (transA == CBlasNoTrans) {
      if (uplo == CBlasUpper) {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t i = 0; i < B->m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
            for (size_t k = 0; k < j; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
          }
        }
      }
      else {
        for (size_t _j = 0; _j < B->n; _j++) {
          size_t j = B->n - _j - 1;
          for (size_t i = 0; i < B->m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
            for (size_t k = j + 1; k < B->n; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, k, j) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
          }
        }
      }
    }
    else {
      if (uplo == CBlasUpper) {
        for (size_t _j = 0; _j < B->n; _j++) {
          size_t j = B->n - _j - 1;
          for (size_t i = 0; i < m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
#ifdef COMPLEX
            if (transA == CBlasTrans) {
#endif
            for (size_t k = j + 1; k < B->n; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, k) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
#ifdef COMPLEX
            }
            else {
              for (size_t k = j + 1; k < B->n; k++) {
                EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, j, k)) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
                EXTENDED total = temp + element;
                carry = (total - temp) - element;
                temp = total;
              }
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
              if (diag == CBlasNonUnit)
                FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / conj((EXTENDED)FUNCTION(matrix, Get)(A, j, j))));
            }
#endif
          }
        }
      }
      else {
        for (size_t j = 0; j < B->n; j++) {
          for (size_t i = 0; i < B->m; i++) {
            EXTENDED temp = 0.l;
            EXTENDED carry = 0.l;
#ifdef COMPLEX
            if (transA == CBlasTrans) {
#endif
            for (size_t k = 0; k < j; k++) {
              EXTENDED element = ((EXTENDED)FUNCTION(matrix, Get)(A, j, k) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
              EXTENDED total = temp + element;
              carry = (total - temp) - element;
              temp = total;
            }
            FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
            if (diag == CBlasNonUnit)
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / (EXTENDED)FUNCTION(matrix, Get)(A, j, j)));
#ifdef COMPLEX
            }
            else {
              for (size_t k = 0; k < j; k++) {
                EXTENDED element = (conj((EXTENDED)FUNCTION(matrix, Get)(A, j, k)) * (EXTENDED)FUNCTION(matrix, Get)(B, i, k)) - carry;
                EXTENDED total = temp + element;
                carry = (total - temp) - element;
                temp = total;
              }
              FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)alpha * (EXTENDED)FUNCTION(matrix, Get)(B, i, j) - temp));
              if (diag == CBlasNonUnit)
                FUNCTION(matrix, Set)(B, i, j, (SCALAR)((EXTENDED)FUNCTION(matrix, Get)(B, i, j) / conj((EXTENDED)FUNCTION(matrix, Get)(A, j, j))));
            }
#endif
          }
        }
      }
    }
  }
}

#undef EXTENDED

#define ERROR_FUNCTION(x) x

static void TEST_FUNCTION(trsm)() {
#define F LINALG_FUNCTION(trsm)
#define PREFIX ""
#include "testtrsm.c"
#undef PREFIX
#undef F
}

static void TEST_FUNCTION2(blas, trsm)() {
#define F LINALG_FUNCTION2(blas, trsm)
#define PREFIX "blas"
#include "testtrsm.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CU_ERROR_CHECK_VOID(x)
#define GPU

static void TEST_FUNCTION2(cu, trsm)() {
#define F LINALG_FUNCTION2(cu, trsm)
#define PREFIX "cu"
#include "testtrsm.c"
#undef PREFIX
#undef F
}

#undef ERROR_FUNCTION
#define ERROR_FUNCTION(x) CUBLAS_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cublas, trsm)() {
#define F LINALG_FUNCTION2(cublas, trsm)
#define PREFIX "cublas"
#define CUBLAS
#include "testtrsm.c"
#undef CUBLAS
#undef PREFIX
#undef F
}

#undef GPU
#undef ERROR_FUNCTION

#ifdef HAS_CULA
#define CULA

#define ERROR_FUNCTION(x) CULA_ERROR_CHECK_VOID(x)

static void TEST_FUNCTION2(cula, trsm)() {
#define F LINALG_FUNCTION2(cula, trsm)
#define PREFIX "cula"
#include "testtrsm.c"
#undef PREFIX
#undef F
}

#define GPU
static void TEST_FUNCTION2(culaDevice, trsm)() {
#define F LINALG_FUNCTION2(culaDevice, trsm)
#define PREFIX "culaDevice"
#include "testtrsm.c"
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

  SCALAR alpha;
  TYPE(matrix) A, B, C;
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&A, max(m + ioff, n + joff), max(m + ioff, n + joff)));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&B, m + ioff, n + joff));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Create)(&C, m + ioff, n + joff));

#ifdef COMPLEX
  alpha = G(LITERAL(0, 0), LITERAL(1, 1));
#else
  alpha = G(LITERAL(0), LITERAL(1));
#endif

  for (size_t j = 0; j < A.n; j++) {
    for (size_t i = 0; i < A.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&A, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

  CONST_TYPE(submatrix) A0 = FUNCTION(matrix, SubmatrixConst)(&A, ioff, ioff, m, m);
  TYPE(submatrix) C0 = FUNCTION(matrix, Submatrix)(&C, ioff, joff, m, n);

#ifdef GPU
  TYPE(CUmatrix) dA, dB;
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Create)(&dA, max(m + ioff, n + joff), max(m + ioff, n + joff)));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Create)(&dB, m + ioff, n + joff));

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dA, &A));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

  CONST_TYPE(CUsubmatrix) dA0 = FUNCTION(cuMatrix, SubmatrixConst)(&dA, ioff, ioff, m, m);
  TYPE(CUsubmatrix) dB0 = FUNCTION(cuMatrix, Submatrix)(&dB, ioff, joff, m, n);

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  TYPE(submatrix) B0 = FUNCTION(matrix, Submatrix)(&B, ioff, joff, m, n);
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUNN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUNU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUTN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUTU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }

#ifdef COMPLEX
  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUCN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUCU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLUCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }
#endif

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLNN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLNU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLTN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLTU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }

#ifdef COMPLEX
  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLCN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA0.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLCU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasLeft, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasLeft, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A0.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmLLCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasLeft, CBlasUnit) / time);
#endif
  }
#endif

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

  CONST_TYPE(submatrix) A1 = FUNCTION(matrix, SubmatrixConst)(&A, joff, joff, n, n);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

  CONST_TYPE(CUsubmatrix) dA1 = FUNCTION(cuMatrix, SubmatrixConst)(&dA, joff, joff, n, n);

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUNN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUNU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUTN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUTU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }

#ifdef COMPLEX
  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUCN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUCU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasUpper, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRUCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }
#endif

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLNN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLNN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLNU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasNoTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLNU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLTN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLTN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLTU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLTU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }

#ifdef COMPLEX
  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLCN,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasNonUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasNonUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLCN,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasNonUnit) / time);
#endif
  }

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++)
#ifdef COMPLEX
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0, 0), LITERAL(1, 1)));
#else
      FUNCTION(matrix, Set)(&B, i, j, G(LITERAL(0), LITERAL(1)));
#endif
  }
  FUNCTION(matrix, Copy)(&C, &B);

#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyHtoD)(&dB, &B));

#ifdef CUBLAS
  ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, CopyDtoH)(&B, &dB));
#else
  ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &B0.m));
#endif
  LINALG_FUNCTION2(gold, trsm)(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &C0.m);

  for (size_t j = 0; j < B.n; j++) {
    for (size_t i = 0; i < B.m; i++) {
#ifdef COMPLEX
      CU_ASSERT_LONG_DOUBLE_EQUAL(creal(FUNCTION(matrix, Get)(&B, i, j)), creal(FUNCTION(matrix, Get)(&B, i, j)), REAL_ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit));
      CU_ASSERT_LONG_DOUBLE_EQUAL(cimag(FUNCTION(matrix, Get)(&B, i, j)), cimag(FUNCTION(matrix, Get)(&B, i, j)), IMAG_ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit));
#else
      CU_ASSERT_LONG_DOUBLE_EQUAL(FUNCTION(matrix, Get)(&B, i, j), FUNCTION(matrix, Get)(&B, i, j), ERROR(i, j, CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit));
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
      ERROR_FUNCTION(F(handle, CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
#ifdef CULA
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m));
#else
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &dA1.m, &dB0.m, NULL));
#endif
#endif
    CU_ERROR_CHECK_VOID(cuEventRecord(stop, 0));
    CU_ERROR_CHECK_VOID(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK_VOID(cuEventElapsedTime(&time, start, stop));
    time /= 1.e3f * (float)iterations;

    CU_ERROR_CHECK_VOID(cuEventDestroy(start));
    CU_ERROR_CHECK_VOID(cuEventDestroy(stop));

    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLCU,%.3e,,%.3e\n", time, (float)FLOPS(CBlasRight, CBlasUnit) / time);
#else
    struct timeval start, stop;
    ERROR_CHECK_VOID(gettimeofday(&start, NULL), (strerror_t)strerror);
    for (size_t i = 0; i < iterations; i++)
      ERROR_FUNCTION(F(CBlasRight, CBlasLower, CBlasConjTrans, CBlasUnit, alpha, &A1.m, &B0.m));
    ERROR_CHECK_VOID(gettimeofday(&stop, NULL), (strerror_t)strerror);
    double time = ((double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.E-6) / (double)iterations;
    fprintf(outfile, PREFIX STRING(BLAS_PREC) "trsmRLCU,%.3e,,%.3e\n", time, (double)FLOPS(CBlasRight, CBlasUnit) / time);
#endif
  }
#endif

  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&A));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&B));
  CU_ERROR_CHECK_VOID(FUNCTION(matrix, Destroy)(&C));
#ifdef GPU
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dA));
  CU_ERROR_CHECK_VOID(FUNCTION(cuMatrix, Destroy)(&dB));
#ifdef CUBLAS
  CUBLAS_ERROR_CHECK_VOID(cublasDestroy(handle));
#endif
#endif
#endif
