#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "error.h"

#define SIZE (128 * 1024 * 1024)
#define ITERATIONS 20

int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  for (int i = 0; i < count; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));

    int memoryClockRate, globalMemoryBusWidth;
    CU_ERROR_CHECK(cuDeviceGetAttribute(&memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

    // Calculate pin bandwidth in bytes/sec (clock rate is actual in kHz, memory is DDR so multiply clock rate by 2.e3 to get effective clock rate in Hz)
    double pinBandwidth = memoryClockRate * 2.e3 * (globalMemoryBusWidth / CHAR_BIT);

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, 0, device));

    fprintf(stdout, "Device %d (pin bandwidth %6.2f GB/s):\n", i, pinBandwidth / (1 << 30));

    CUDA_MEMCPY2D copy;
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    CUevent start, stop;
    CU_ERROR_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_ERROR_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    float time;

    // Calculate aligned copy for 32, 64 and 128-bit word sizes
    for (int j = 4; j <= 16; j *= 2) {
      copy.WidthInBytes = SIZE;
      copy.Height = 1;

      copy.srcXInBytes = 0;
      copy.srcY = 0;
      copy.dstXInBytes = 0;
      copy.dstY = 0;

      CU_ERROR_CHECK(cuMemAllocPitch(&copy.srcDevice, &copy.srcPitch, copy.srcXInBytes + copy.WidthInBytes, copy.Height, j));
      CU_ERROR_CHECK(cuMemAllocPitch(&copy.dstDevice, &copy.dstPitch, copy.dstXInBytes + copy.WidthInBytes, copy.Height, j));

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (size_t j = 0; j < ITERATIONS; j++)
        CU_ERROR_CHECK(cuMemcpy2D(&copy));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));
      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= ITERATIONS * 1.e3f;
      double bandwidth = 2 * copy.WidthInBytes * copy.Height / time;

      fprintf(stdout, "\taligned copy (%3u-bit): %6.2f GB/s (%5.2f%%)\n", j * CHAR_BIT, bandwidth / (1 << 30), (bandwidth / pinBandwidth) * 100.0);

      CU_ERROR_CHECK(cuMemFree(copy.srcDevice));
      CU_ERROR_CHECK(cuMemFree(copy.dstDevice));
    }

    // Calculate misaligned copy for 32, 64 and 128-bit word sizes
    for (int j = 4; j <= 16; j *= 2) {
      copy.WidthInBytes = SIZE;
      copy.Height = 1;

      copy.srcXInBytes = j;
      copy.srcY = 0;
      copy.dstXInBytes = j;
      copy.dstY = 0;

      CU_ERROR_CHECK(cuMemAllocPitch(&copy.srcDevice, &copy.srcPitch, copy.srcXInBytes + copy.WidthInBytes, copy.Height, j));
      CU_ERROR_CHECK(cuMemAllocPitch(&copy.dstDevice, &copy.dstPitch, copy.dstXInBytes + copy.WidthInBytes, copy.Height, j));

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (size_t j = 0; j < ITERATIONS; j++)
        CU_ERROR_CHECK(cuMemcpy2D(&copy));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));
      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= ITERATIONS * 1.e3f;
      double bandwidth = 2 * copy.WidthInBytes * copy.Height / time;

      fprintf(stdout, "\tmisaligned copy (%3u-bit): %6.2f GB/s (%5.2f%%)\n", j * CHAR_BIT, bandwidth / (1 << 30), (bandwidth / pinBandwidth) * 100.0);

      CU_ERROR_CHECK(cuMemFree(copy.srcDevice));
      CU_ERROR_CHECK(cuMemFree(copy.dstDevice));
    }

    // Calculate stride-2 copy for 32, 64 and 128-bit word sizes
    for (int j = 4; j <= 16; j *= 2) {
      copy.WidthInBytes = SIZE / 2;
      copy.Height = 1;

      copy.srcXInBytes = 0;
      copy.srcY = 0;
      copy.dstXInBytes = 0;
      copy.dstY = 0;

      CU_ERROR_CHECK(cuMemAllocPitch(&copy.srcDevice, &copy.srcPitch, copy.srcXInBytes + copy.WidthInBytes, copy.Height, j));
      CU_ERROR_CHECK(cuMemAllocPitch(&copy.dstDevice, &copy.dstPitch, copy.dstXInBytes + copy.WidthInBytes, copy.Height, j));

      copy.srcPitch *= 2;
      copy.dstPitch *= 2;

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (size_t j = 0; j < ITERATIONS; j++)
        CU_ERROR_CHECK(cuMemcpy2D(&copy));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));
      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= ITERATIONS * 1.e3f;
      double bandwidth = 2 * copy.WidthInBytes * copy.Height / time;

      fprintf(stdout, "\tstride-2 copy (%3u-bit): %6.2f GB/s (%5.2f%%)\n", j * CHAR_BIT, bandwidth / (1 << 30), (bandwidth / pinBandwidth) * 100.0);

      CU_ERROR_CHECK(cuMemFree(copy.srcDevice));
      CU_ERROR_CHECK(cuMemFree(copy.dstDevice));
    }

    // Calculate stride-10 copy for 32, 64 and 128-bit word sizes
    for (int j = 4; j <= 16; j *= 2) {
      copy.WidthInBytes = SIZE / 10;
      copy.Height = 1;

      copy.srcXInBytes = 0;
      copy.srcY = 0;
      copy.dstXInBytes = 0;
      copy.dstY = 0;

      CU_ERROR_CHECK(cuMemAllocPitch(&copy.srcDevice, &copy.srcPitch, copy.srcXInBytes + copy.WidthInBytes, copy.Height, j));
      CU_ERROR_CHECK(cuMemAllocPitch(&copy.dstDevice, &copy.dstPitch, copy.dstXInBytes + copy.WidthInBytes, copy.Height, j));

      copy.srcPitch *= 10;
      copy.dstPitch *= 10;

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (size_t j = 0; j < ITERATIONS; j++)
        CU_ERROR_CHECK(cuMemcpy2D(&copy));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));
      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= ITERATIONS * 1.e3f;
      double bandwidth = 2 * copy.WidthInBytes * copy.Height / time;

      fprintf(stdout, "\tstride-10 copy (%3u-bit): %6.2f GB/s (%5.2f%%)\n", j * CHAR_BIT, bandwidth / (1 << 30), (bandwidth / pinBandwidth) * 100.0);

      CU_ERROR_CHECK(cuMemFree(copy.srcDevice));
      CU_ERROR_CHECK(cuMemFree(copy.dstDevice));
    }

    // Calculate stride-1000 copy for 32, 64 and 128-bit word sizes
    for (int j = 4; j <= 16; j *= 2) {
      copy.WidthInBytes = SIZE / 1000;
      copy.Height = 1;

      copy.srcXInBytes = 0;
      copy.srcY = 0;
      copy.dstXInBytes = 0;
      copy.dstY = 0;

      CU_ERROR_CHECK(cuMemAllocPitch(&copy.srcDevice, &copy.srcPitch, copy.srcXInBytes + copy.WidthInBytes, copy.Height, j));
      CU_ERROR_CHECK(cuMemAllocPitch(&copy.dstDevice, &copy.dstPitch, copy.dstXInBytes + copy.WidthInBytes, copy.Height, j));

      copy.srcPitch *= 1000;
      copy.dstPitch *= 1000;

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (size_t j = 0; j < ITERATIONS; j++)
        CU_ERROR_CHECK(cuMemcpy2D(&copy));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));
      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= ITERATIONS * 1.e3f;
      double bandwidth = 2 * copy.WidthInBytes * copy.Height / time;

      fprintf(stdout, "\tstride-1000 copy (%3u-bit): %6.2f GB/s (%5.2f%%)\n", j * CHAR_BIT, bandwidth / (1 << 30), (bandwidth / pinBandwidth) * 100.0);

      CU_ERROR_CHECK(cuMemFree(copy.srcDevice));
      CU_ERROR_CHECK(cuMemFree(copy.dstDevice));
    }

    CU_ERROR_CHECK(cuEventDestroy(start));
    CU_ERROR_CHECK(cuEventDestroy(stop));

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
