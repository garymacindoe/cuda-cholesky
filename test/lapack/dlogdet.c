#include "lapack.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>

int main(int argc, char * argv[]) {
  size_t n;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <n>\nwhere:\n"
                    "  n  is the size of the matrix\n", argv[0]);
    return 1;
  }

  if (sscanf(argv[1], "%zu", &n) != 1) {
    fprintf(stderr, "Unable to parse number from '%s'\n", argv[1]);
    return 1;
  }

  srand(0);

  double * x;
  size_t incx = 1;

  if ((x = malloc(incx *  n * sizeof(double))) == NULL) {
    fputs("Unable to allocate x\n", stderr);
    return -1;
  }

  for (size_t j = 0; j < n; j++)
    x[j * incx] = ((double)rand() + 1.0) / (double)RAND_MAX;

  double res = dlogdet(x, incx, n);

  double sum = 0.0;
  double c = 0.0;
  for (size_t j = 0; j < n; j++) {
    double y = log(x[j * incx]) - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  sum *= 2.0;

  double diff = fabs(sum - res);
  bool passed = (diff < 2.0 * (double)n * DBL_EPSILON);

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    dlogdet(x, incx, n);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  const size_t bandwidth = n * sizeof(double);
  fprintf(stdout, "%.3es %.3gGB/s Error: %.3e\n%sED!\n", time,
          (double)bandwidth / (time * (double)(1 << 30)), diff, (passed) ? "PASS" : "FAIL");

  free(x);

  return (int)!passed;
}
