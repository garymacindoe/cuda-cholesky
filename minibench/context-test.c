#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>
#include "error.h"

// The number of contexts that can be created on a GPU depends on the memory
// available on the GPU (which will be less if there is a display attached).
// This works for on a 1GB GTX 285 with a display attached (and can be up to 20
// on one without) but needs to be a lot lower on a 128MB NVS 140M.
#define N 10

int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  CUdevice devices[count];
  for (int i = 0; i < count; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  CUcontext contexts[count][N];
  for (int i = 0; i < count; i++) {
    struct timeval start, stop;

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuCtxCreate(&contexts[i][j], CU_CTX_SCHED_AUTO, devices[i]));
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxCreate (device %d): %.3es\n", i, time / (double)N);
  }

  fprintf(stderr, "\n");

  for (int i = 0; i < count; i++) {
    struct timeval start, stop;

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[i][j]));
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxPopCurrent (device %d): %.3es\n", i, time / (double)N);
  }

  fprintf(stderr, "\n");

  if (count > 1) {
    struct timeval start, stop;

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++) {
      for (int i = 0; i < count; i++)
        CU_ERROR_CHECK(cuCtxPushCurrent(contexts[i][j]));
    }
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxPushCurrent (alternating devices): %.3es\n", time / (double)N);

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++) {
      for (int i = 0; i < count; i++)
        CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[i][j]));
    }
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxPopCurrent (alternating devices): %.3es\n\n", time / (double)N);
  }

  for (int i = 0; i < count; i++) {
    struct timeval start, stop;

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[i][j]));
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxPushCurrent (device %d): %.3es\n", i, time / (double)N);
  }

  fprintf(stderr, "\n");

  for (int i = 0; i < count; i++) {
    struct timeval start, stop;

    ERROR_CHECK(gettimeofday(&start, 0), (strerror_t)strerror);
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuCtxDestroy(contexts[i][j]));
    ERROR_CHECK(gettimeofday(&stop, 0), (strerror_t)strerror);

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuCtxDestroy (device %d): %.3es\n", i, time / (double)N);
  }

  return 0;
}
