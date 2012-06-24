#include <cuda.h>
#include <stdio.h>

#include "error.h"

#define ITERATIONS 1E6

#ifndef CUBINDIR
#define CUBINDIR "cubin"
#endif

/**
 * This measures the overhead in launching a kernel function on each GPU in the
 * system.
 *
 * It does this by executing a small kernel (copying 1 value in global memory) a
 * very large number of times and taking the average execution time.  This
 * program uses the CUDA driver API.
 */
int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  float x = 5.0f;
  for (int d = 0; d < count; d++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, d));

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, 0, device));

    CUdeviceptr in, out;
    CU_ERROR_CHECK(cuMemAlloc(&in, sizeof(float)));
    CU_ERROR_CHECK(cuMemAlloc(&out, sizeof(float)));
    CU_ERROR_CHECK(cuMemcpyHtoD(in, &x, sizeof(float)));

    CUmodule module;
    CU_ERROR_CHECK(cuModuleLoad(&module, "kernel-test.ptx"));

    CUfunction function;
    CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "kernel"));

    void * params[] = { &in, &out };

    CUevent start, stop;
    CU_ERROR_CHECK(cuEventCreate(&start, 0));
    CU_ERROR_CHECK(cuEventCreate(&stop, 0));

    CU_ERROR_CHECK(cuEventRecord(start, 0));
    for (int i = 0; i < ITERATIONS; i++)
      CU_ERROR_CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, params, NULL));

    CU_ERROR_CHECK(cuEventRecord(stop, 0));
    CU_ERROR_CHECK(cuEventSynchronize(stop));

    float time;
    CU_ERROR_CHECK(cuEventElapsedTime(&time, start, stop));

    CU_ERROR_CHECK(cuEventDestroy(start));
    CU_ERROR_CHECK(cuEventDestroy(stop));

    CU_ERROR_CHECK(cuMemFree(in));
    CU_ERROR_CHECK(cuMemFree(out));

    fprintf(stdout, "Device %d: %fms\n", d, (time / (double)ITERATIONS));

    CU_ERROR_CHECK(cuModuleUnload(module));

    CU_ERROR_CHECK(cuCtxDestroy(context));
  }

  return 0;
}
