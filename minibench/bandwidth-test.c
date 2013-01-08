#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "error.h"

#define SIZE (128 * 1024 * 1024)
#define INCREMENT (1024 * 1024)
#define ITERATIONS 20

/**
 * This measures the bandwidth obtained when transferring data between CPU
 * memory and GPU memory across the PCI-Express bus.
 *
 * Contiguous areas of memory ranging from 1MB to 128MB (in 1MB increments) are
 * copied a large number of times.  The average time for each size of transfer
 * is taken.  A least squares regression is then performed to calculate the
 * average bandwidth and the overhead in setting up the memory transfer.
 *
 * The experiment is repeat for host to device transfers and device to host
 * transfers for each GPU in the system.
 */
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
    CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

    CUstream stream;
    CU_ERROR_CHECK(cuStreamCreate(&stream, 0));

    CUdeviceptr dPointer;
    CU_ERROR_CHECK(cuMemAlloc(&dPointer, SIZE));

    void * hPointer;
    CU_ERROR_CHECK(cuMemAllocHost(&hPointer, SIZE));

    double sumX = 0.0, sumXX = 0.0, sumY = 0.0, sumXY = 0.0;
    size_t n = SIZE / INCREMENT;
    for (size_t j = 1; j <= n; j++) {
      size_t size = j * INCREMENT;
      struct timeval start, stop;

      int error;
      if ((error = gettimeofday(&start, NULL)) != 0) {
        fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
        return error;
      }
      for (size_t k = 0; k < ITERATIONS; k++)
        CU_ERROR_CHECK(cuMemcpyHtoDAsync(dPointer, hPointer, size, stream));
      CU_ERROR_CHECK(cuStreamSynchronize(stream));
      if ((error = gettimeofday(&stop, NULL)) != 0) {
        fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
        return error;
      }
      double time = ((double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6)) / (double)ITERATIONS;

      sumX += (double)size;
      sumXX += (double)size * (double)size;
      sumY += time;
      sumXY += time * (double)size;
    }
    double sxx = sumXX - (sumX * sumX) / (double)n;
    double sxy = sumXY - (sumX * sumY) / (double)n;
    double xbar = sumX / (double)n;
    double ybar = sumY / (double)n;
    double m = sxy / sxx;
    fprintf(stdout, "\tHost to Device: %4.2fMB/s + %3.3fms\n", 1.0 / (m * (1 << 20)), (ybar - m * xbar) * 1.E3);

    sumY = 0.0, sumXY = 0.0;
    for (size_t j = 1; j <= n; j++) {
      size_t size = j * INCREMENT;
      struct timeval start, stop;

      int error;
      if ((error = gettimeofday(&start, NULL)) != 0) {
        fprintf(stderr, "Unable to get start time: %s\n", strerror(error));
        return error;
      }
      for (size_t k = 0; k < ITERATIONS; k++)
        CU_ERROR_CHECK(cuMemcpyDtoHAsync(hPointer, dPointer, size, stream));
      CU_ERROR_CHECK(cuStreamSynchronize(stream));
      if ((error = gettimeofday(&stop, NULL)) != 0) {
        fprintf(stderr, "Unable to get stop time: %s\n", strerror(error));
        return error;
      }
      double time = ((double)(stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec) * 1.E-6)) / (double)ITERATIONS;

      sumY += time;
      sumXY += time * (double)size;
    }
    sxx = sumXX - (sumX * sumX) / (double)n;
    sxy = sumXY - (sumX * sumY) / (double)n;
    xbar = sumX / (double)n;
    ybar = sumY / (double)n;
    m = sxy / sxx;
    fprintf(stdout, "\tDevice to Host: %4.2fMB/s + %3.3fms\n", 1.0 / (m * (1 << 20)), (ybar - m * xbar) * 1.E3);

    CU_ERROR_CHECK(cuMemFreeHost(hPointer));

    CU_ERROR_CHECK(cuMemFree(dPointer));

    CU_ERROR_CHECK(cuStreamDestroy(stream));

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
