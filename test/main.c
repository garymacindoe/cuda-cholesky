#include <CUnit/CUnit.h>
#include <CUnit/Automated.h>
#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include <CUnit/CUCurses.h>
#include <getopt.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "error.h"

#ifdef USE_CULA
#include <cula.h>
#endif

CU_ErrorCode registerSuites();
int getTestOptions(const struct option **, const char **);
int processTestOption(int, int, const char *, int, int);
int printTestOptions(FILE *);
static void CU_errorHandler(const char *, const char *, const char *, unsigned int, int, strerror_t);

static int display_help(FILE *, const char *);

static inline bool isnull(struct option opt) {
  return opt.name == NULL && opt.has_arg == 0 && opt.flag == NULL && opt.val == 0;
}

static inline size_t optlen(const struct option * longopts) {
  size_t len = 0; while (!isnull(longopts[len])) len++; return len;
}

static inline struct option * optcpy(struct option * dest, const struct option * src) {
  struct option * res = dest; while (!isnull(*dest++ = *src++)); return res;
}

int main(int argc, char * argv[]) {
  int error;

  const char * suite = NULL, * test = NULL;
  bool list = false, help = false, type_set = false;

  enum run_type { AUTO, BASIC, CONSOLE, CURSES };
  enum run_type run = BASIC;

  struct option global_longopts[] = { { "suite",   required_argument,         NULL, 's' },
                                      { "test",    required_argument,         NULL, 't' },
                                      { "auto",    optional_argument,         NULL,  2  },
                                      { "basic",   optional_argument,         NULL,  3  },
                                      { "console",       no_argument,         NULL,  4  },
                                      { "curses",        no_argument,         NULL,  5  },
                                      { "list",          no_argument, (int *)&list,  1  },
                                      { "help",          no_argument, (int *)&help,  1  },
                                      { NULL,                    0,         NULL,  0  } };
  const char * global_optstring = "+:s:t:lh";

  const struct option * test_longopts;
  const char * test_optstring;
  if ((error = getTestOptions(&test_longopts, &test_optstring)) != 0) {
    fprintf(stderr, "Unable to get test options\n");
    return error;
  }

  size_t global_longopts_len = optlen(global_longopts);
  size_t global_optstring_len = strlen(global_optstring);
  size_t test_longopts_len = optlen(test_longopts);
  size_t test_optstring_len = strlen(test_optstring);

  struct option * longopts;
  char * optstring;
  if ((longopts = (struct option *)malloc((global_longopts_len + test_longopts_len + 1) * sizeof(struct option))) == NULL) {
    fprintf(stderr, "failed to allocate memory for long options\n");
    return -1;
  }
  if ((optstring = (char *)malloc((global_optstring_len + test_optstring_len + 1) * sizeof(struct option))) == NULL) {
    fprintf(stderr, "failed to allocate memory for long options\n");
    return -1;
  }

  optcpy(longopts, global_longopts);
  optcpy(longopts + global_longopts_len, test_longopts);
  strcpy(optstring, global_optstring);
  strcpy(optstring + global_optstring_len, test_optstring);

  opterr = 0;
  int c, longindex;
  while ((c = getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
    switch (c) {
      case 's':
        if (optarg == NULL) {
          fprintf(stderr, "Option \"--%s\" requires an argument\n", longopts[longindex].name);
          return -optind - 1;
        }
        suite = optarg;
        break;
      case 't':
        if (optarg == NULL) {
          fprintf(stderr, "Option \"--%s\" requires an argument\n", longopts[longindex].name);
          return -optind - 1;
        }
        test = optarg;
        break;
      case 2:
        if (type_set) {
          fprintf(stderr, "Run type has already been specified\n");
          return -optind - 1;
        }
        run = AUTO;
        if (optarg != NULL)
          CU_set_output_filename(optarg);
        type_set = true;
        break;
      case 3:
        if (type_set) {
          fprintf(stderr, "Run type has already been specified\n");
          return -optind - 1;
        }
        run = BASIC;
        if (optarg != NULL) {
          if(strncmp(optarg, "normal", 6) == 0)
            CU_basic_set_mode(CU_BRM_NORMAL);
          else if (strncmp(optarg, "silent", 7) == 0)
            CU_basic_set_mode(CU_BRM_SILENT);
          else if (strncmp(optarg, "verbose", 7) == 0)
            CU_basic_set_mode(CU_BRM_VERBOSE);
          else {
            fprintf(stderr, "Unknown basic run mode \"%s\"\n", optarg);
            return -optind - 1;
          }
        }
        type_set = true;
        break;
      case 4:
        if (type_set) {
          fprintf(stderr, "Run type has already been specified\n");
          return -optind - 1;
        }
        run = CONSOLE;
        type_set = true;
        break;
      case 5:
        if (type_set) {
          fprintf(stderr, "Run type has already been specified\n");
          return -optind - 1;
        }
        run = CURSES;
        type_set = true;
        break;
      case 'l': list = true; break;
      case 'h': help = true; break;
      case 0: break;
      case ':':
        fprintf(stderr, "Option '-%c' requires an argument\n", (char)optopt);
        return -1;
      default:
        if ((error = processTestOption(c, longindex - (int)global_longopts_len, optarg, optind, optopt)) < 0) {
          if (error == -1) {
            fprintf(stderr, "Unknown option '-%c'\n", (char)optopt);
            display_help(stderr, argv[0]);
          }
          return error;
        }
    }
  }

  if (suite == NULL && test != NULL) {
    fprintf(stderr, "Option --test requires option --suite\n");
    return -1;
  }

  if (help) {
    display_help(stdout, argv[0]);
    return 0;
  }

  CU_ErrorCode cuerror;
  if ((cuerror = CU_initialize_registry()) != CUE_SUCCESS) {
    fprintf(stderr, "failed to initialise test registry: %s\n", CU_get_error_msg());
    return cuerror;
  }

  if ((cuerror = registerSuites()) != CUE_SUCCESS) {
    fprintf(stderr, "failed to register test suites: %s\n", CU_get_error_msg());
    return cuerror;
  }

  CU_pTestRegistry registry = CU_get_registry();

  if (list) {
    for (CU_pSuite suite = registry->pSuite; suite != NULL; suite = suite->pNext) {
      fprintf(stdout, "%s\n", suite->pName);
      for (CU_pTest test = suite->pTest; test != NULL; test = test->pNext)
        fprintf(stdout, "  %s\n", test->pName);
    }
    CU_cleanup_registry();
    return 0;
  }

  CUresult result;
  if ((result = cuInit(0)) != CUDA_SUCCESS) {
    fprintf(stderr, "Error initialising CUDA runtime: %s\n", cuGetErrorString(result));
    return result;
  }

  errorHandler = CU_errorHandler;

  if (run == AUTO)
    CU_automated_run_tests();
  else if (run == BASIC) {
    if (suite != NULL) {
      CU_pSuite pSuite = NULL;
      CU_pTest pTest = NULL;

      for (CU_pSuite s = registry->pSuite; s != NULL && pSuite == NULL; s = s->pNext) {
        if (strcmp(s->pName, suite) == 0)
          pSuite = s;
      }

      if (pSuite == NULL) {
        fprintf(stderr, "No suite named \"%s\" could be found\n", suite);
        return 1;
      }

      if (test != NULL) {
        for (CU_pTest t = pSuite->pTest; t != NULL && pTest == NULL; t = t->pNext) {
          if (strcmp(t->pName, test) == 0)
            pTest = t;
        }

        if (pTest == NULL) {
          fprintf(stderr, "No test named \"%s\" could be found in suite \"%s\"\n", test, suite);
          return 1;
        }
      }

      cuerror = (pTest == NULL) ? CU_basic_run_suite(pSuite) : CU_basic_run_test(pSuite, pTest);
    }
    else
      cuerror = CU_basic_run_tests();

    if (cuerror != CUE_SUCCESS) {
      fprintf(stderr, "failed to run tests: %s\n", CU_get_error_msg());
      return cuerror;
    }

    CU_basic_show_failures(CU_get_failure_list());
    if (CU_get_number_of_failure_records() > 0) printf("\n");
  }
  else if (run == CONSOLE)
    CU_console_run_tests();
  else
    CU_curses_run_tests();

  unsigned int failures = CU_get_number_of_tests_failed();

  CU_cleanup_registry();

  return (int)failures;
}

static int display_help(FILE * f, const char * name) {
  return fprintf(f, "Usage: %s [global_options] [test_options]\n", name) +
  fprintf(f, "Global Options:\n") +
  fprintf(f, "  -s, --suite=<suitename>  run the specified suite (default: run all suites)\n") +
  fprintf(f, "  -t, --test=<testname>    run the specified test within the suite (default: run all tests within the suite)\n") +
  fprintf(f, "  -l, --list               list the tests available to run\n") +
  fprintf(f, "  -h, --help               display this help\n") +
  fprintf(f, "Test Options:\n") +
  printTestOptions(f);
}

static void CU_errorHandler(const char * call, const char * function, const char * file, unsigned int line, int error, strerror_t strerror) {
  char condition[512];
  if (strerror == NULL)
    snprintf(condition, 512, "%s returned %d", call, error);
  else
    snprintf(condition, 512, "%s returned %d (%s)", call, error, strerror(error));
  CU_assertImplementation(CU_FALSE, line, condition, file, function, CU_TRUE);
}
