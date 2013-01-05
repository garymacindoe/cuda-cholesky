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

  for (int i = 0; i < count; i++) {
    CUstream streams[N];
    struct timeval start, stop;

    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

    int error;
    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuStreamCreate(&streams[j], 0));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuStreamCreate (device %d): %.3es\n", i, time / (double)N);

    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuStreamDestroy(streams[j]));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "cuStreamDestroy (device %d): %.3es\n", i, time / (double)N);

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
