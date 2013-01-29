#include "cumultigpu.h"
#include <stdio.h>
#include <assert.h>
#include "error.h"

// Print out info on GPU in background thread
CUresult print(const void * args) {
  (void)args;

  CUdevice device;
  CU_ERROR_CHECK(cuCtxGetDevice(&device));

  char name[256];
  CU_ERROR_CHECK(cuDeviceGetName(name, 256, device));

  size_t bytes;
  CU_ERROR_CHECK(cuDeviceTotalMem(&bytes, device));

  fprintf(stdout, "Using device %s with %zuMB total memory\n", name, bytes / (1 << 20));

  return CUDA_SUCCESS;
}

int main() {
  CU_ERROR_CHECK(cuInit(0));

  CUdevice device;
  CU_ERROR_CHECK(cuDeviceGet(&device, 0));

  CUmultiGPU mGPU;
  CU_ERROR_CHECK(cuMultiGPUCreate(&mGPU, &device, 1));

  assert(cuMultiGPUGetContextCount(mGPU) == 1);

  CUtask task;
  CU_ERROR_CHECK(cuTaskCreate(&task, print, NULL, 0));

  assert(cuMultiGPURunTask(mGPU, 1, task) == CUDA_ERROR_INVALID_VALUE);
  assert(cuMultiGPURunTask(mGPU, 0, task) == CUDA_SUCCESS);

  CU_ERROR_CHECK(cuMultiGPUSynchronize(mGPU));

  CUresult result;
  CU_ERROR_CHECK(cuTaskDestroy(task, &result));

  assert(result == CUDA_SUCCESS);

  CU_ERROR_CHECK(cuMultiGPUDestroy(mGPU));

  return 0;
}
