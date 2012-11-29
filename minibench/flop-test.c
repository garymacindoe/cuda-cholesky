#include <cuda.h>
#include <stdio.h>

#include "error.h"

#define ITERATIONS 20
#define DUAL_ISSUE

/**
 * This measures the single precision floating point arithmetic throughput of
 * each GPU in the system.
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

    int multiProcessorCount, clockRate, maxThreadsPerBlock, major, minor;
    CU_ERROR_CHECK(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    CU_ERROR_CHECK(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
    CU_ERROR_CHECK(cuDeviceComputeCapability(&major, &minor, device));

    /**
     * Throughput for single precision multiply and multiply-add is 8
     * instructions per multiprocessor per clock cycle for devices of compute
     * capability 1.x, 32 for compute capability 2.0 and 48 for compute
     * capability 2.1.
     *
     * Multiply-add is 2 FLOPs.
     *
     * nVidia CUDA Programming Guide v4.1 Section 5.4.1
     */
    double theoreticalGFlops = (major == 2 ? (minor == 0 ? 32 : 48) : 8) * 2 * multiProcessorCount * clockRate * 1.E-6;

    CUmodule module;
    CU_ERROR_CHECK(cuModuleLoad(&module, "flop-test.fatbin"));

    CUdeviceptr data;
    CU_ERROR_CHECK(cuMemAlloc(&data, (size_t)maxThreadsPerBlock * sizeof(float)));

    union { float f; unsigned int j; } fint;
    fint.f = 1.0f;
    CU_ERROR_CHECK(cuMemsetD32(data, fint.j, (size_t)maxThreadsPerBlock));

    CUfunction function;
    CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "fmad"));

    void * params[] = { &data };

    CUevent start, stop;
    CU_ERROR_CHECK(cuEventCreate(&start, 0));
    CU_ERROR_CHECK(cuEventCreate(&stop, 0));

    CU_ERROR_CHECK(cuEventRecord(start, 0));
    for (int i = 0; i < ITERATIONS; i++)
      CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)multiProcessorCount, 1, 1, (unsigned int)maxThreadsPerBlock, 1, 1, 0, 0, params, NULL));

    CU_ERROR_CHECK(cuEventRecord(stop, 0));
    CU_ERROR_CHECK(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
    time /= (float)ITERATIONS;

    double actualGFlops = (131072u * multiProcessorCount * maxThreadsPerBlock * 1.E-6) / time;

    fprintf(stdout, "Device %d: %3.2f / %3.2f GFlops/s (%2.2f%%, fmaf)\n", d, actualGFlops, theoreticalGFlops, (actualGFlops / theoreticalGFlops) * 100.0);

    /**
     * Each multiprocessor also has a number of single precision special
     * function units (2, 16 and 4 for compute capability 1.x, 2.0 and 2.1
     * respectively).  These can also perform single precision floating point
     * multiplications (1 FLOP, 8 instructions per multiprocessor per clock
     * cycle) in parallel with the single precision scalar processors.
     *
     * nVidia CUDA Programming Guide v4.1 Sections F.3.1 and F.4.1
     */
    theoreticalGFlops += (major == 2 ? (minor == 0 ? 16 : 4) : 1) * 8 * multiProcessorCount * clockRate * 1.E-6;

    CU_ERROR_CHECK(cuMemsetD32(data, fint.j, (size_t)maxThreadsPerBlock));

    CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "fmadmul"));

    CU_ERROR_CHECK(cuEventRecord(start, 0));
    for (int i = 0; i < ITERATIONS; i++)
      CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)multiProcessorCount, 1, 1, (unsigned int)maxThreadsPerBlock, 1, 1, 0, 0, params, NULL));

    CU_ERROR_CHECK(cuEventRecord(stop, 0));
    CU_ERROR_CHECK(cuEventSynchronize(stop));

    CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
    time /= (float)ITERATIONS;

    CU_ERROR_CHECK(cuEventDestroy(start));
    CU_ERROR_CHECK(cuEventDestroy(stop));

    CU_ERROR_CHECK(cuMemFree(data));

    actualGFlops = (196608u * multiProcessorCount * maxThreadsPerBlock * 1.E-6) / time;

    fprintf(stdout, "           %3.2f / %3.2f GFlops/s (%2.2f%%, fmaf + fmulf)\n", actualGFlops, theoreticalGFlops, (actualGFlops / theoreticalGFlops) * 100.0);

    if (major >= 1 && minor >= 2) {

    /**
     * Throughput for double precision multiply-add is 1 instruction per
     * multiprocessor per clock cycle for devices of compute capability 1.x, 16
     * for compute capability 2.0 and 4 for compute capability 2.1.
     *
     * Multiply-add is 2 FLOPs.
     *
     * nVidia CUDA Programming Guide v4.1 Section 5.4.1 Table 5.1
     */
      double theoreticalGFlops = (major == 2 ? (minor == 0 ? 16 : 4) : 1) * 2 * multiProcessorCount * clockRate * 1.E-6;

      CU_ERROR_CHECK(cuMemAlloc(&data, (size_t)maxThreadsPerBlock * sizeof(double)));

      union { double d; unsigned long m; } dint;
      dint.d = 1.0;
      CU_ERROR_CHECK(cuMemsetD32(data, (unsigned int)dint.m, (size_t)maxThreadsPerBlock * 2));

      CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "dmad"));

      void * params[] = { &data };

      CU_ERROR_CHECK(cuEventCreate(&start, 0));
      CU_ERROR_CHECK(cuEventCreate(&stop, 0));

      CU_ERROR_CHECK(cuEventRecord(start, 0));
      for (int i = 0; i < ITERATIONS; i++)
      CU_ERROR_CHECK(cuLaunchKernel(function, (unsigned int)multiProcessorCount, 1, 1, (unsigned int)maxThreadsPerBlock, 1, 1, 0, 0, params, NULL));
      CU_ERROR_CHECK(cuEventRecord(stop, 0));
      CU_ERROR_CHECK(cuEventSynchronize(stop));

      CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));
      time /= (float)ITERATIONS;

      CU_ERROR_CHECK(cuEventDestroy(start));
      CU_ERROR_CHECK(cuEventDestroy(stop));

      CU_ERROR_CHECK(cuMemFree(data));

      actualGFlops = (131072u * multiProcessorCount * maxThreadsPerBlock * 1.E-6) / time;

      fprintf(stdout, "          %3.2f / %3.2f GFlops/s (%2.2f%% fma)\n", actualGFlops, theoreticalGFlops, (actualGFlops / theoreticalGFlops) * 100.0);
    }

    CU_ERROR_CHECK(cuModuleUnload(module));

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
