#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>
#include "error.h"

#define N 20
#define SIZE 1024

int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  for (int i = 0; i < count; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));

    char name[256];
    CU_ERROR_CHECK(cuDeviceGetName(name, 256, device));

    size_t bytes;
    CU_ERROR_CHECK(cuDeviceTotalMem(&bytes, device));

    int major, minor;
    CU_ERROR_CHECK(cuDeviceComputeCapability(&major, &minor, device));

    fprintf(stdout, "Device %d (%s, CC %d.%d, %zuMB):\n", i, name, major, minor, bytes >> 20);

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_BLOCKING_SYNC, device));

    CUdeviceptr ptrs[N];
    struct timeval start, stop;
    int error;
    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuMemAlloc(&ptrs[j], SIZE));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    double time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "  cuMemAlloc: %.3es\n", time / (double)N);

    if ((error = gettimeofday(&start, NULL)) != 0) {
      fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
      return error;
    }
    for (size_t j = 0; j < N; j++)
      CU_ERROR_CHECK(cuMemFree(ptrs[j]));
    if ((error = gettimeofday(&stop, NULL)) != 0) {
      fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
      return error;
    }

    time = (double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6);
    fprintf(stderr, "  cuMemFree : %.3es\n", time / (double)N);

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
