#include <CUnit/CUnit.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>

#include "error.h"
#include "blas.h"
#include "gaussian.h"

#define PROCESS_REQUIRED_OPTION(id, var, format, desc) \
 case id: \
   if (optarg == NULL) { \
     fprintf(stderr, "Option \"--%s\" requires an argument\n", options[longindex].name); \
     return -optind - 1; \
   } \
   if ((sscanf(optarg, format, &var)) != 1) { \
     fprintf(stderr, "Unable to parse " desc " from \"%s\"\n", optarg); \
     return -optind - 1; \
   } \
   break;

static int ordinals[] = { 0, 1, 2, 3 };         // GPUs to use for tests
static size_t nGPUs = 1;
static size_t m = 64, n = 64, k = 64;           // Size of matrix to use when benchmarking (may be 0)
static size_t ioff = 7, joff = 7, koff = 7;     // Submatrix offsets to use
static size_t iterations = 20;                  // Number of iterations to perform to measure bandwidth/FLOPs
static FILE * outfile = NULL;                   // Where to output benchmark results (may be NULL)
static unsigned int seed = 1234u;               // Seed for random number generator

static const struct option options[] = {
  { "devices",       required_argument, NULL, 'd' },
  { "iterations",    required_argument, NULL, 'i' },
  { "column-offset", required_argument, NULL, 200 },
  { "row-offset",    required_argument, NULL, 201 },
  { "inner-offset",  required_argument, NULL, 202 },
  { "benchmark",     required_argument, NULL, 'b' },
  { "seed",          required_argument, NULL, 203 },
  { NULL,            0,                 NULL,  0  }
};

int getTestOptions(const struct option ** longopts, const char ** optstring) {
  *longopts = options;
  *optstring = "d:i:m:n:k:b:";

  return 0;
}

int processTestOption(int c, int longindex, const char * optarg, int optind, int optopt) {
  (void)optopt;
  switch (c) {
    case 'd':
      if (optarg == NULL) {
        fprintf(stderr, "Option \"%s\" requires an argument\n", options[longindex].name);
        return -optind - 1;
      }
      for (nGPUs = 0; nGPUs < 4 && *optarg != '\0'; nGPUs++) {
        if (sscanf(optarg, "%d", &ordinals[nGPUs]) != 1) {
          fprintf(stderr, "Unable to parse device ordinals from \"%s\"\n", optarg);
          return -optind - 1;
        }
        do { optarg++; } while (*optarg != ',' && *optarg != '\0');
      }
      break;
    PROCESS_REQUIRED_OPTION('i', iterations, "%zu", "number of benchmark iterations");
    PROCESS_REQUIRED_OPTION('m', m, "%zu", "matrix column length");
    PROCESS_REQUIRED_OPTION('n', n, "%zu", "matrix row length");
    PROCESS_REQUIRED_OPTION('k', k, "%zu", "matrix inner length");
    PROCESS_REQUIRED_OPTION(200, ioff, "%zu", "submatrix column offset");
    PROCESS_REQUIRED_OPTION(201, joff, "%zu", "submatrix row offset");
    PROCESS_REQUIRED_OPTION(202, koff, "%zu", "submatrix inner offset");
    PROCESS_REQUIRED_OPTION(203, seed, "%u", "seed");
    case 'b':
      if (optarg == NULL) {
        fprintf(stderr, "Option \"--%s\" requires an argument\n", options[longindex].name);
        return -optind - 1;
      }
      if (strncmp(optarg, "-", 1) == 0)
        outfile = stdout;
      else if ((outfile = fopen(optarg, "w")) == NULL) {
        fprintf(stderr, "Unable to open \"%s\" for writing\n", optarg);
        return -optind - 1;
      }
      break;
    default:
      return -1;
  }
  return 0;
}

int printTestOptions(FILE * f) {
  return fprintf(f,
                 "  -d, --devices=<d>       run the tests on the specified devices. Single GPU tests will run on the first GPU (default: all available devices)\n"
                 "  -i, --iterations=<n>    run each test for <n> iterations when measuring performance (default: 20)\n"
                 "  -m=<m>                  use a matrix with <m> rows in each column for tests (default: 64)\n"
                 "  -n=<n>                  use a matrix with <n> columns in each row for tests (default: 64)\n"
                 "  -k=<k>                  use a matrix with <k> columns or rows as the second argument for tests (default: 64)\n"
                 "      --column-offset=<i> use a submatrix with a column offset of <i> rows (default: 7)\n"
                 "      --row-offset=<j>    use a submatrix with a row offset of <j> columns (default: 7)\n"
                 "      --inner-offset=<k>  use a submatrix with an inner offset of <k> columns (default: 7)\n"
                 "      --seed=<s>          seed the system random number generator with <s> before using it to initialise test matrices (default: 1234)\n"
                 "  -b, --benchmark=<file>  output benchmark result to <file> (default: don't run benchmarks)\n");
}

static inline size_t max(size_t a, size_t b) { return (a > b) ? a : b; }
static inline size_t min(size_t a, size_t b) { return (a < b) ? a : b; }

#define TEST_FUNCTION(name) CONCAT3(test, BLAS_PREC, name)
#define TEST_FUNCTION2(type, name) CONCAT4(test, type, BLAS_PREC2, name)
#define TEST(name) { STRING(BLAS_PREC) #name, TEST_FUNCTION(name) }
#define TEST2(type, name) { #type STRING(BLAS_PREC2) #name, TEST_FUNCTION2(type, name) }
#define TEST_ARRAY() CONCAT2(BLAS_PREC, Test)
#define TEST_ARRAY2(type) CONCAT3(type, BLAS_PREC2, Test)

#define FLOAT_T

#define BLAS_PREC s
#define BLAS_PREC2 S
#define G gaussianf

#include "templates_on.h"
#include "testgemm.c"
#include "testsyrk.c"
#include "testtrmm.c"
#include "testtrsm.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(gemm),
  TEST(syrk),
  TEST(trmm),
  TEST(trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(blas)[] = {
  TEST2(blas, gemm),
  TEST2(blas, syrk),
  TEST2(blas, trmm),
  TEST2(blas, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, gemm),
  TEST2(cu, syrk),
  TEST2(cu, trmm),
  TEST2(cu, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, gemm),
  TEST2(cuMultiGPU, syrk),
  TEST2(cuMultiGPU, trmm),
  TEST2(cuMultiGPU, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cublas)[] = {
  TEST2(cublas, gemm),
  TEST2(cublas, syrk),
  TEST2(cublas, trmm),
  TEST2(cublas, trsm),
  CU_TEST_INFO_NULL
};

#include "templates_off.h"

#undef G
#undef BLAS_PREC2
#undef BLAS_PREC

#undef FLOAT_T

#define DOUBLE_T

#define BLAS_PREC d
#define BLAS_PREC2 D
#define G gaussian

#include "templates_on.h"
#include "testgemm.c"
#include "testsyrk.c"
#include "testtrmm.c"
#include "testtrsm.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(gemm),
  TEST(syrk),
  TEST(trmm),
  TEST(trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(blas)[] = {
  TEST2(blas, gemm),
  TEST2(blas, syrk),
  TEST2(blas, trmm),
  TEST2(blas, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, gemm),
  TEST2(cu, syrk),
  TEST2(cu, trmm),
  TEST2(cu, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, gemm),
  TEST2(cuMultiGPU, syrk),
  TEST2(cuMultiGPU, trmm),
  TEST2(cuMultiGPU, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cublas)[] = {
  TEST2(cublas, gemm),
  TEST2(cublas, syrk),
  TEST2(cublas, trmm),
  TEST2(cublas, trsm),
  CU_TEST_INFO_NULL
};

#include "templates_off.h"

#undef G
#undef BLAS_PREC2
#undef BLAS_PREC

#undef DOUBLE_T

#define FLOAT_COMPLEX_T

#define BLAS_PREC c
#define BLAS_PREC2 C
#define G cgaussianf

#include "templates_on.h"
#include "testgemm.c"
#include "testherk.c"
#include "testtrmm.c"
#include "testtrsm.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(gemm),
  TEST(herk),
  TEST(trmm),
  TEST(trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(blas)[] = {
  TEST2(blas, gemm),
  TEST2(blas, herk),
  TEST2(blas, trmm),
  TEST2(blas, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, gemm),
  TEST2(cu, herk),
  TEST2(cu, trmm),
  TEST2(cu, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, gemm),
  TEST2(cuMultiGPU, herk),
  TEST2(cuMultiGPU, trmm),
  TEST2(cuMultiGPU, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cublas)[] = {
  TEST2(cublas, gemm),
  TEST2(cublas, herk),
  TEST2(cublas, trmm),
  TEST2(cublas, trsm),
  CU_TEST_INFO_NULL
};

#include "templates_off.h"

#undef G
#undef BLAS_PREC2
#undef BLAS_PREC

#undef FLOAT_COMPLEX_T

#define DOUBLE_COMPLEX_T

#define BLAS_PREC z
#define BLAS_PREC2 Z
#define G cgaussian

#include "templates_on.h"
#include "testgemm.c"
#include "testherk.c"
#include "testtrmm.c"
#include "testtrsm.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(gemm),
  TEST(herk),
  TEST(trmm),
  TEST(trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(blas)[] = {
  TEST2(blas, gemm),
  TEST2(blas, herk),
  TEST2(blas, trmm),
  TEST2(blas, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, gemm),
  TEST2(cu, herk),
  TEST2(cu, trmm),
  TEST2(cu, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, gemm),
  TEST2(cuMultiGPU, herk),
  TEST2(cuMultiGPU, trmm),
  TEST2(cuMultiGPU, trsm),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cublas)[] = {
  TEST2(cublas, gemm),
  TEST2(cublas, herk),
  TEST2(cublas, trmm),
  TEST2(cublas, trsm),
  CU_TEST_INFO_NULL
};

#include "templates_off.h"

#undef G
#undef BLAS_PREC2
#undef BLAS_PREC

#undef DOUBLE_COMPLEX_T

#undef TEST_FUNCTION
#undef TEST_FUNCTION2
#undef TEST
#undef TEST2
#undef TEST_ARRAY
#undef TEST_ARRAY2

static CUdevice devices[4];
static CUcontext * contexts;

static int testSetUp() {
  srand(seed);
  return 0;
}

static int cuSuiteInit() {
  for (size_t i = 0; i < nGPUs; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], ordinals[i]));
  return CUDA_SUCCESS;
}

static int cuTestSetUp() {
  if ((contexts == malloc(nGPUs * sizeof(CUcontext))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < nGPUs; i++)
    CU_ERROR_CHECK(cuCtxCreate(&contexts[i], 0, devices[i]));
  srand(seed);
  return CUDA_SUCCESS;
}

static int cuTestTearDown() {
  for (size_t i = 0; i < nGPUs; i++)
    CU_ERROR_CHECK(cuCtxDestroy(contexts[i]));
  free(contexts);
  return CUDA_SUCCESS;
}

CU_ErrorCode registerSuites() {
  static CU_SuiteInfo suites[] = {
    { "sblas3", NULL, NULL, testSetUp, NULL, sTest },
    { "dblas3", NULL, NULL, testSetUp, NULL, dTest },
    { "cblas3", NULL, NULL, testSetUp, NULL, cTest },
    { "zblas3", NULL, NULL, testSetUp, NULL, zTest },
    { "blasSblas3", NULL, NULL, testSetUp, NULL, blasSTest },
    { "blasDblas3", NULL, NULL, testSetUp, NULL, blasDTest },
    { "blasCblas3", NULL, NULL, testSetUp, NULL, blasCTest },
    { "blasZblas3", NULL, NULL, testSetUp, NULL, blasZTest },
    { "cuSblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuSTest },
    { "cuDblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuDTest },
    { "cuCblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuCTest },
    { "cuZblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuZTest },
    { "cuMultiGPUSblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUSTest },
    { "cuMultiGPUDblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUDTest },
    { "cuMultiGPUCblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUCTest },
    { "cuMultiGPUZblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUZTest },
    { "cublasSblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cublasSTest },
    { "cublasDblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cublasDTest },
    { "cublasCblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cublasCTest },
    { "cublasZblas3", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cublasZTest },
    CU_SUITE_INFO_NULL
  };

  return CU_register_suites(suites);
}
