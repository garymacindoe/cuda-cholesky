#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "error.h"

#define EXPECTED_GPU_ALIGNMENT 256u
#define EXPECTED_CPU_ALIGNMENT 16u

/**
 * A simple benchmark to prove that host and device pointers allocated through
 * CUDA are aligned on 256 byte boundaries (also in 2D via cudaMallocPitch) and
 * that pointers from malloc are aligned on 16 byte boundaries (for MKL and
 * vectorising).
 */
int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  for (int i = 0; i < count; i++) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, i));

    CUcontext context;
    CU_ERROR_CHECK(cuCtxCreate(&context, 0, device));

    for (size_t j = 1; j < 4096; j++) {
      CUdeviceptr dPointer;
      size_t pitch;
      CU_ERROR_CHECK(cuMemAllocPitch(&dPointer, &pitch, j * sizeof(float), 2, sizeof(float)));

      void * hPointer;
      CU_ERROR_CHECK(cuMemAllocHost(&hPointer, j * sizeof(float)));

      void * h = malloc(j * sizeof(float));

      if (((size_t)dPointer) % EXPECTED_GPU_ALIGNMENT != 0 || pitch % EXPECTED_GPU_ALIGNMENT != 0) {
        fprintf(stdout, "GPU %d alignment is not %u\n", i, EXPECTED_GPU_ALIGNMENT);
        return 1;
      }
      else if (((size_t)hPointer) % EXPECTED_CPU_ALIGNMENT != 0 || ((size_t)h) % EXPECTED_CPU_ALIGNMENT != 0) {
        fprintf(stdout, "CPU alignment is not %u\n", EXPECTED_CPU_ALIGNMENT);
        return 1;
      }

      free(h);
      CU_ERROR_CHECK(cuMemFreeHost(hPointer));
      CU_ERROR_CHECK(cuMemFree(dPointer));
    }

    CU_ERROR_CHECK(cuCtxDestroy(context));

  }

  fprintf(stdout, "GPU alignment is at least %u bytes\nCPU alignment is at least %u bytes\n", EXPECTED_GPU_ALIGNMENT, EXPECTED_CPU_ALIGNMENT);

  return 0;
}
