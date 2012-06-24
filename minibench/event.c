#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

#include "error.h"

/**
 * This benchmark queries how each GPU in the system handles event timing.
 * It records an event on the GPU, sleeps for a second on the CPU, then records
 * a stop event on the GPU.
 *
 * If the GPU measures the time between two events as a number of clock ticks
 * spent executing the GPU program then the time between the two events will be
 * less than a second (as there is no GPU code executed between the events).
 * This is the same method used as calling clock() twice and dividing the
 * difference by CLOCKS_PER_SEC on the CPU.
 *
 * If the GPU measures the time between two events by recording a timestamp in
 * each event then the time measured between the two events will be equal to or
 * greater than a second.  This is the same method used as calling
 * clock_gettime() or gettimeofday() twice and taking the difference.  The
 * accuracy of this method in timing program execution is dependent on the load
 * on the system.  On the CPU this matters but not on the GPU as the GPU cannot
 * multitask and suspends graphics operations while executing a GPGPU program.
 */
int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  for (int d = 0; d < count; d++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, d));

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, 0, device));

    CUevent start, stop;
    CU_ERROR_CHECK(cuEventCreate(&start, 0));
    CU_ERROR_CHECK(cuEventCreate(&stop, 0));

    CU_ERROR_CHECK(cuEventRecord(start, 0));
    sleep(1);
    CU_ERROR_CHECK(cuEventRecord(stop, 0));
    CU_ERROR_CHECK(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));

    CU_ERROR_CHECK(cuEventDestroy(start));
    CU_ERROR_CHECK(cuEventDestroy(stop));

    fprintf(stdout, "GPU (%d) event timing done by %s\n", d, (time < 1.0f) ? "ticks" : "timestamp");

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
