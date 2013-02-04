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

  float * x;
  size_t incx = 1;

  if ((x = malloc(incx *  n * sizeof(float))) == NULL) {
    fputs("Unable to allocate x\n", stderr);
    return -1;
  }

  for (size_t j = 0; j < n; j++)
    x[j * incx] = ((float)rand() + 1.0f) / (float)RAND_MAX;

  float res = slogdet(x, incx, n);

  float sum = 0.0f;
  float c = 0.0f;
  for (size_t j = 0; j < n; j++) {
    float y = logf(x[j * incx]) - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  sum *= 2.0f;

  float diff = fabsf(sum - res);
  bool passed = (diff < 2.0f * (float)n * FLT_EPSILON);

  struct timeval start, stop;
  if (gettimeofday(&start, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -4;
  }
  for (size_t i = 0; i < 20; i++)
    slogdet(x, incx, n);
  if (gettimeofday(&stop, NULL) != 0) {
    fprintf(stderr, "gettimeofday failed at %s:%d\n", __FILE__, __LINE__);
    return -5;
  }

  double time = ((double)(stop.tv_sec - start.tv_sec) +
                 (double)(stop.tv_usec - start.tv_usec) * 1.e-6) / 20.0;
  const size_t bandwidth = n * sizeof(float);
  fprintf(stdout, "%.3es %.3gGB/s Error: %.3e\n%sED!\n", time,
          (double)bandwidth / (time * (double)(1 << 30)), diff, (passed) ? "PASS" : "FAIL");

  free(x);

  return (int)!passed;
}
