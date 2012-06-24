#include <stdio.h>
#include <time.h>
#include <string.h>

#include "error.h"

int main() {
  struct timespec res;
  ERROR_CHECK(clock_getres(CLOCK_MONOTONIC, &res), (strerror_t)strerror);
  fprintf(stdout, "Timer resolution: %.3fns\n", (double)res.tv_sec * 10E9 + (double)res.tv_nsec);
  return 0;
}
