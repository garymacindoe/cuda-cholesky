#include <CUnit/CUnit.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <tgmath.h>

#include "error.h"
#include "lapack.h"
#include "blas.h"
#include "gaussian.h"
#include "multigpu.h"

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

static int ordinals[] = { 0, 1, 2, 3 }; // GPUs to use for tests
static size_t nGPUs = 1;
static size_t n = 64;           // Size of matrix to use when benchmarking (may be 0)
static size_t ioff = 7, joff = 7;       // Submatrix offsets to use
static size_t iterations = 20;          // Number of iterations to perform to measure bandwidth/FLOPs
static FILE * outfile = NULL;           // Where to output benchmark results (may be NULL)
static unsigned int seed = 1234u;       // Seed for random number generator

static const struct option options[] = {
  { "devices",       required_argument, NULL, 'd' },
  { "iterations",    required_argument, NULL, 'i' },
  { "benchmark",     required_argument, NULL, 'b' },
  { "column-offset", required_argument, NULL, 200 },
  { "row-offset",    required_argument, NULL, 201 },
  { "seed",          required_argument, NULL, 202 },
  { NULL,            0,                 NULL,  0  }
};

int getTestOptions(const struct option ** longopts, const char ** optstring) {
  *longopts = options;
  *optstring = "d:i:n:b:";

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
    PROCESS_REQUIRED_OPTION('n', n, "%zu", "matrix size");
    PROCESS_REQUIRED_OPTION(200, ioff, "%zu", "submatrix column offset");
    PROCESS_REQUIRED_OPTION(201, joff, "%zu", "submatrix row offset");
    PROCESS_REQUIRED_OPTION(202, seed, "%u", "seed");
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
                 "  -d, --device=<d>        run the tests on the specified device (default: 0)\n"
                 "  -i, --iterations=<n>    run each test for <n> iterations when measuring performance (default: 20)\n"
                 "  -n=<n>                  use a square <n>x<n> matrix for tests (default: 64)\n"
                 "      --column-offset=<i> use a submatrix with a column offset of <i> rows (default: 7)\n"
                 "      --row-offset=<j>    use a submatrix with a row offset of <j> columns (default: 7)\n"
                 "      --seed=<s>          seed the system random number generator with <s> before using it to initialise test matrices (default: 1234)\n"
                 "  -b, --benchmark=<file>  output benchmark result to <file> (default: don't run benchmarks)\n");
}

#define TEST_FUNCTION(name) CONCAT3(test, LAPACK_PREC, name)
#define TEST_FUNCTION2(type, name) CONCAT4(test, type, LAPACK_PREC2, name)
#define TEST(name) { STRING(LAPACK_PREC) #name, TEST_FUNCTION(name) }
#define TEST2(type, name) { #type STRING(LAPACK_PREC2) #name, TEST_FUNCTION2(type, name) }
#define TEST_ARRAY() CONCAT2(LAPACK_PREC, Test)
#define TEST_ARRAY2(type) CONCAT3(type, LAPACK_PREC2, Test)

#define FLOAT_T

#define LAPACK_PREC s
#define LAPACK_PREC2 S
#define G gaussianf

#include "templates_on.h"
#include "testlauu2.c"
#include "testlauum.c"
#include "testpotf2.c"
#include "testpotrf.c"
#include "testpotri.c"
#include "testtrti2.c"
#include "testtrtri.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(lauu2),
  TEST(lauum),
  TEST(potf2),
  TEST(potrf),
  TEST(potri),
  TEST(trti2),
  TEST(trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(lapack)[] = {
  TEST2(lapack, lauu2),
  TEST2(lapack, lauum),
  TEST2(lapack, potf2),
  TEST2(lapack, potrf),
  TEST2(lapack, potri),
  TEST2(lapack, trti2),
  TEST2(lapack, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, lauum),
  TEST2(cuMultiGPU, potrf),
  TEST2(cuMultiGPU, potri),
  TEST2(cuMultiGPU, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, lauu2),
  TEST2(cu, potf2),
  TEST2(cu, potrf),
  TEST2(cu, potri),
  TEST2(cu, trti2),
  CU_TEST_INFO_NULL
};

#ifdef HAS_CULA
static CU_TestInfo TEST_ARRAY2(cula)[] = {
  TEST2(cula, potrf),
  TEST2(cula, potri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(culaDevice)[] = {
  TEST2(culaDevice, potrf),
  TEST2(culaDevice, potri),
  CU_TEST_INFO_NULL
};
#endif

#include "templates_off.h"

#undef G
#undef LAPACK_PREC2
#undef LAPACK_PREC

#undef FLOAT_T

#define DOUBLE_T

#define LAPACK_PREC d
#define LAPACK_PREC2 D
#define G gaussian

#include "templates_on.h"
#include "testlauu2.c"
#include "testlauum.c"
#include "testpotf2.c"
#include "testpotrf.c"
#include "testpotri.c"
#include "testtrti2.c"
#include "testtrtri.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(lauu2),
  TEST(lauum),
  TEST(potf2),
  TEST(potrf),
  TEST(potri),
  TEST(trti2),
  TEST(trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(lapack)[] = {
  TEST2(lapack, lauu2),
  TEST2(lapack, lauum),
  TEST2(lapack, potf2),
  TEST2(lapack, potrf),
  TEST2(lapack, potri),
  TEST2(lapack, trti2),
  TEST2(lapack, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, lauum),
  TEST2(cuMultiGPU, potrf),
  TEST2(cuMultiGPU, potri),
  TEST2(cuMultiGPU, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, lauu2),
  TEST2(cu, potf2),
  TEST2(cu, potrf),
  TEST2(cu, potri),
  TEST2(cu, trti2),
  CU_TEST_INFO_NULL
};

#ifdef HAS_CULA
static CU_TestInfo TEST_ARRAY2(cula)[] = {
  TEST2(cula, potrf),
  TEST2(cula, potri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(culaDevice)[] = {
  TEST2(culaDevice, potrf),
  TEST2(culaDevice, potri),
  CU_TEST_INFO_NULL
};
#endif

#include "templates_off.h"

#undef G
#undef LAPACK_PREC2
#undef LAPACK_PREC

#undef DOUBLE_T

#define FLOAT_COMPLEX_T

#define LAPACK_PREC c
#define LAPACK_PREC2 C
#define G cgaussianf

#include "templates_on.h"
#include "testlacgv.c"
#include "testlauu2.c"
#include "testlauum.c"
#include "testpotf2.c"
#include "testpotrf.c"
#include "testpotri.c"
#include "testtrti2.c"
#include "testtrtri.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(lacgv),
  TEST(lauu2),
  TEST(lauum),
  TEST(potf2),
  TEST(potrf),
  TEST(potri),
  TEST(trti2),
  TEST(trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(lapack)[] = {
  TEST2(lapack, lacgv),
  TEST2(lapack, lauu2),
  TEST2(lapack, lauum),
  TEST2(lapack, potf2),
  TEST2(lapack, potrf),
  TEST2(lapack, potri),
  TEST2(lapack, trti2),
  TEST2(lapack, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, lauum),
  TEST2(cuMultiGPU, potrf),
  TEST2(cuMultiGPU, potri),
  TEST2(cuMultiGPU, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, lacgv),
  TEST2(cu, lauu2),
  TEST2(cu, potf2),
  TEST2(cu, potrf),
  TEST2(cu, potri),
  TEST2(cu, trti2),
  CU_TEST_INFO_NULL
};

#ifdef HAS_CULA
static CU_TestInfo TEST_ARRAY2(cula)[] = {
  TEST2(cula, potrf),
  TEST2(cula, potri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(culaDevice)[] = {
  TEST2(culaDevice, potrf),
  TEST2(culaDevice, potri),
  CU_TEST_INFO_NULL
};
#endif

#include "templates_off.h"

#undef G
#undef LAPACK_PREC2
#undef LAPACK_PREC

#undef FLOAT_COMPLEX_T

#define DOUBLE_COMPLEX_T

#define LAPACK_PREC z
#define LAPACK_PREC2 Z
#define G cgaussian

#include "templates_on.h"
#include "testlacgv.c"
#include "testlauu2.c"
#include "testlauum.c"
#include "testpotf2.c"
#include "testpotrf.c"
#include "testpotri.c"
#include "testtrti2.c"
#include "testtrtri.c"

static CU_TestInfo TEST_ARRAY()[] = {
  TEST(lacgv),
  TEST(lauu2),
  TEST(lauum),
  TEST(potf2),
  TEST(potrf),
  TEST(potri),
  TEST(trti2),
  TEST(trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(lapack)[] = {
  TEST2(lapack, lacgv),
  TEST2(lapack, lauu2),
  TEST2(lapack, lauum),
  TEST2(lapack, potf2),
  TEST2(lapack, potrf),
  TEST2(lapack, potri),
  TEST2(lapack, trti2),
  TEST2(lapack, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cuMultiGPU)[] = {
  TEST2(cuMultiGPU, lauum),
  TEST2(cuMultiGPU, potrf),
  TEST2(cuMultiGPU, potri),
  TEST2(cuMultiGPU, trtri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(cu)[] = {
  TEST2(cu, lacgv),
  TEST2(cu, lauu2),
  TEST2(cu, potf2),
  TEST2(cu, potrf),
  TEST2(cu, potri),
  TEST2(cu, trti2),
  CU_TEST_INFO_NULL
};

#ifdef HAS_CULA
static CU_TestInfo TEST_ARRAY2(cula)[] = {
  TEST2(cula, potrf),
  TEST2(cula, potri),
  CU_TEST_INFO_NULL
};

static CU_TestInfo TEST_ARRAY2(culaDevice)[] = {
  TEST2(culaDevice, potrf),
  TEST2(culaDevice, potri),
  CU_TEST_INFO_NULL
};
#endif

#include "templates_off.h"

#undef G
#undef LAPACK_PREC2
#undef LAPACK_PREC

#undef DOUBLE_COMPLEX_T

#undef TEST_FUNCTION
#undef TEST_FUNCTION2
#undef TEST
#undef TEST2
#undef TEST_ARRAY
#undef TEST_ARRAY2

static CUdevice devices[4];
static CUmultiGPU mGPU;

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
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, nGPUs, CU_CTX_SCHED_AUTO, devices));
  srand(seed);
  return CUDA_SUCCESS;
}

static int cuTestTearDown() {
  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));
  return CUDA_SUCCESS;
}

#ifdef HAS_CULA
static int culaSuiteInit() {
  CULA_ERROR_CHECK(culaSelectDevice(ordinals[0]));
  return culaNoError;
}

static int culaTestSetUp() {
  CULA_ERROR_CHECK(culaInitialize());
  srand(seed);
  return culaNoError;
}

static int culaTestTearDown() {
  culaShutdown();
  return culaNoError;
}
#endif

CU_ErrorCode registerSuites() {
  static CU_SuiteInfo suites[] = {
    { "slapack", NULL, NULL, testSetUp, NULL, sTest },
    { "dlapack", NULL, NULL, testSetUp, NULL, dTest },
    { "clapack", NULL, NULL, testSetUp, NULL, cTest },
    { "zlapack", NULL, NULL, testSetUp, NULL, zTest },
    { "lapackSlapack", NULL, NULL, testSetUp, NULL, lapackSTest },
    { "lapackDlapack", NULL, NULL, testSetUp, NULL, lapackDTest },
    { "lapackClapack", NULL, NULL, testSetUp, NULL, lapackCTest },
    { "lapackZlapack", NULL, NULL, testSetUp, NULL, lapackZTest },
    { "cuSlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuSTest },
    { "cuDlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuDTest },
    { "cuClapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuCTest },
    { "cuZlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuZTest },
    { "cuMultiGPUSlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUSTest },
    { "cuMultiGPUDlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUDTest },
    { "cuMultiGPUClapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUCTest },
    { "cuMultiGPUZlapack", cuSuiteInit, NULL, cuTestSetUp, cuTestTearDown, cuMultiGPUZTest },
#ifdef HAS_CULA
    { "culaSlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaSTest },
    { "culaDlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaDTest },
    { "culaClapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaCTest },
    { "culaZlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaZTest },
    { "culaDeviceSlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaDeviceSTest },
    { "culaDeviceDlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaDeviceDTest },
    { "culaDeviceClapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaDeviceCTest },
    { "culaDeviceZlapack", culaSuiteInit, NULL, culaTestSetUp, culaTestTearDown, culaDeviceZTest },
#endif
    CU_SUITE_INFO_NULL
  };

  return CU_register_suites(suites);
}
