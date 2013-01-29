#include <cuda.h>
#include <stdio.h>
#include <limits.h>

#include "error.h"

/**
 * This calculates the theoretical flop:word ratio of each GPU in the system.
 */
int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  for (int d = 0; d < count; d++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, d));

    int major, minor, multiProcessorCount, clockRate, memoryClockRate, globalMemoryBusWidth;
    CU_ERROR_CHECK(cuDeviceComputeCapability(&major, &minor, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

    long bandwidth = memoryClockRate * 2000 * globalMemoryBusWidth / CHAR_BIT;
    long flops = (major == 2 ? (minor == 0 ? 32 : 48) : 8) * multiProcessorCount * clockRate * 2000;

    fprintf(stdout, "Device %d: %3.2f GFlops/s single precision fmad, %3.2f GB/s bandwidth, flop:word %2.2f\n", d, (double)flops * 1.e-9, (double)bandwidth / (double)(1 << 30), (double)flops / ((double)bandwidth / 4.0));

    if (major >= 1 && minor >= 2) {
      flops = (major == 2 ? (minor == 0 ? 16 : 4) : 1) * multiProcessorCount * clockRate * 2000;

      fprintf(stdout, "           %3.2f GFlops/s double precision dmad, %3.2f GB/s bandwidth, flop:word %2.2f\n", (double)flops * 1.e-9, (double)bandwidth / (double)(1 << 30), (double)flops / ((double)bandwidth / 8.0));
    }
  }

  return 0;
}
