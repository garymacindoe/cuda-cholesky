#include <stdio.h>
#include <time.h>
#include <string.h>

int main() {
  int error;
  struct timespec res;
  if ((error = clock_getres(CLOCK_MONOTONIC, &res)) != 0) {
    fprintf(stderr, "Unable to get clock resolution: %s\n", strerror(error));
    return error;
  }
  fprintf(stdout, "Timer resolution: %.3fns\n", (double)res.tv_sec * 10E9 + (double)res.tv_nsec);
  return 0;
}
