#include <cuda.h>
#include <stdio.h>
#include "error.h"

#define N 2048

int main() {
  CU_ERROR_CHECK(cuInit(0));

  int count;
  CU_ERROR_CHECK(cuDeviceGetCount(&count));

  count = (count > 2) ? 2 : count;

  CUdevice devices[count];
  for (int i = 0; i < count; i++)
    CU_ERROR_CHECK(cuDeviceGet(&devices[i], i));

  // Question 1:  Can you create multiple contexts on the same device?
  {
    fprintf(stderr, "Attempting to create multiple contexts on each device...\n");
    CUcontext contexts[count * N];
    size_t j = 0;
    for (int i = 0; i < count; i++) {
      CUresult error = CUDA_SUCCESS;
      size_t k;
      for (k = 0; k < N && error == CUDA_SUCCESS; k++) {
        error = cuCtxCreate(&contexts[j], CU_CTX_SCHED_AUTO, devices[i]);
        if (error == CUDA_SUCCESS)
          CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[j++]));
      }
      fprintf(stderr, "  created %zu contexts on device %d before cuCtxCreate returned \"%s\"\n", (k - 1), i, cuGetErrorString(error));
    }

    CUresult error = CUDA_SUCCESS;
    size_t k;
    for (k = 0; k < j && error == CUDA_SUCCESS; k++)
      error = cuCtxPushCurrent(contexts[k]);
    if (error == CUDA_SUCCESS)
      fprintf(stderr, "  successfully pushed %zu contexts with cuCtxPushCurrent\n", k);
    else
      fprintf(stderr, "  pushed %zu contexts before cuCtxPushCurrent returned \"%s\"\n", (k - 1), cuGetErrorString(error));

    for (size_t k = 0; k < j; k++)
      CU_ERROR_CHECK(cuCtxDestroy(contexts[k]));

    fprintf(stderr, "\n");
  }

  CUcontext contexts[count][2];
  for (int i = 0; i < count; i++) {
    for (size_t j = 0; j < 2; j++) {
      CU_ERROR_CHECK(cuCtxCreate(&contexts[i][j], CU_CTX_SCHED_AUTO, devices[i]));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[i][j]));
    }
  }

  // Question 2:  Can you access a host pointer in a different context from
  // which it was created?
  // Question 3:  Can you free a host pointer in a different context from which
  // it was created?
  {
    void * hPtr;
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
    CU_ERROR_CHECK(cuMemAllocHost(&hPtr, 1024));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));

    CUdeviceptr dPtr[count];
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
    CU_ERROR_CHECK(cuMemAlloc(&dPtr[0], 1024)); // Different context, same device
    fprintf(stderr, "Accessing a host pointer from a different context to which it was allocated (on the same device) returns \"%s\"\n", cuGetErrorString(cuMemcpyHtoD(dPtr[0], hPtr, 1024)));
    CU_ERROR_CHECK(cuMemFree(dPtr[0]));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    if (count > 1) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      CU_ERROR_CHECK(cuMemAlloc(&dPtr[1], 1024)); // Different context, different device
      fprintf(stderr, "Accessing a host pointer from a different context to which it was allocated (on a different device) returns \"%s\"\n", cuGetErrorString(cuMemcpyHtoD(dPtr[1], hPtr, 1024)));
    CU_ERROR_CHECK(cuMemFree(dPtr[1]));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }

    fprintf(stderr, "\n");

    CUresult error = CUDA_ERROR_UNKNOWN;
    if (count > 1) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      error = cuMemFreeHost(hPtr);
      fprintf(stderr, "Freeing a host pointer from a different context to which it was allocated (on a different device) returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
      error = cuMemFreeHost(hPtr);
      fprintf(stderr, "Freeing a host pointer from a different context to which it was allocated (on the same device) returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
      error = cuMemFreeHost(hPtr);
      fprintf(stderr, "Freeing a host pointer from the same context to which it was allocated returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    }

    fprintf(stderr, "\n");
  }

  // Question 4:  Can you access a device pointer in a different context from
  // which it was created?
  // Question 5:  Can you free a device pointer in a different context from which
  // it was created?
  {
    CUdeviceptr dPtr[count][2];
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
    CU_ERROR_CHECK(cuMemAlloc(&dPtr[0][0], 1024));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
    CU_ERROR_CHECK(cuMemAlloc(&dPtr[0][1], 1024));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));

    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
    fprintf(stderr, "Accessing a device pointer from a different context to which it was allocated (on the same device) returns \"%s\"\n", cuGetErrorString(cuMemcpyDtoD(dPtr[0][0], dPtr[0][1], 1024)));
    CU_ERROR_CHECK(cuMemFree(dPtr[0][1]));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));

    if (count > 1) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      CU_ERROR_CHECK(cuMemAlloc(&dPtr[1][0], 1024)); // Different context, different device
      fprintf(stderr, "Accessing a device pointer from a different context to which it was allocated (on a different device) returns \"%s\"\n", cuGetErrorString(cuMemcpyDtoD(dPtr[0][0], dPtr[1][0], 1024)));
      CU_ERROR_CHECK(cuMemFree(dPtr[1][0]));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }

    fprintf(stderr, "\n");

    CUresult error = CUDA_ERROR_UNKNOWN;
    if (count > 1) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      error = cuMemFree(dPtr[0][0]);
      fprintf(stderr, "Freeing a device pointer from a different context to which it was allocated (on a different device) returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
      error = cuMemFree(dPtr[0][0]);
      fprintf(stderr, "Freeing a device pointer from a different context to which it was allocated (on the same device) returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
      error = cuMemFree(dPtr[0][0]);
      fprintf(stderr, "Freeing a device pointer from the same context to which it was allocated returns \"%s\"\n", cuGetErrorString(error));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    }

    fprintf(stderr, "\n");
  }

  // Question 6:  Can you access a module in a different context from which it
  // was loaded?
  // Question 7:  Can you unload a module in a different context from which it
  // was loaded?
  {
    CUmodule module;
    CUdeviceptr ptr;
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
    CU_ERROR_CHECK(cuModuleLoad(&module,  "kernel-test.ptx"));
    CU_ERROR_CHECK(cuMemAlloc(&ptr, sizeof(float)));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));

    CUfunction function = 0;
    if (count > 0) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      fprintf(stderr, "Getting a function pointer from a different context to which it was loaded (on a different device) returns \"%s\"\n", cuGetErrorString(cuModuleGetFunction(&function, module, "kernel")));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }
    if (function == 0) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
      fprintf(stderr, "Getting a function pointer from a different context to which it was loaded (on the same device) returns \"%s\"\n", cuGetErrorString(cuModuleGetFunction(&function, module, "kernel")));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    }
    if (function == 0) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
      fprintf(stderr, "Getting a function pointer from the same context to which it was loaded returns \"%s\"\n", cuGetErrorString(cuModuleGetFunction(&function, module, "kernel")));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    }

    fprintf(stderr, "\n");

    CUdeviceptr a, b;
    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
    CU_ERROR_CHECK(cuMemAlloc(&a, sizeof(float)));
    CU_ERROR_CHECK(cuMemAlloc(&b, sizeof(float)));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    void * params[] = { &a, & b };

    CUresult error = CUDA_ERROR_UNKNOWN;
    if (count > 0) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      fprintf(stderr, "Launching a function from a different context to which it was loaded (on a different device) returns \"%s\"\n", cuGetErrorString(error = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, params, NULL)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
      fprintf(stderr, "Launching a function from a different context to which it was loaded (on the same device) returns \"%s\"\n", cuGetErrorString(error = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, params, NULL)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
      fprintf(stderr, "Launching a function from the same context to which it was loaded returns \"%s\"\n", cuGetErrorString(error = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, params, NULL)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    }

    fprintf(stderr, "\n");

    error = CUDA_ERROR_UNKNOWN;
    if (count > 0) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[1][0]));
      fprintf(stderr, "Unloading a module from a different context to which it was loaded (on a different device) returns \"%s\"\n", cuGetErrorString(error = cuModuleUnload(module)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[1][0]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][1]));
      fprintf(stderr, "Unloading a module from a different context to which it was loaded (on the same device) returns \"%s\"\n", cuGetErrorString(error = cuModuleUnload(module)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][1]));
    }
    if (error != CUDA_SUCCESS) {
      CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
      fprintf(stderr, "Unloading a module from the same context to which it was loaded returns \"%s\"\n", cuGetErrorString(error = cuModuleUnload(module)));
      CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
    }

    CU_ERROR_CHECK(cuCtxPushCurrent(contexts[0][0]));
    CU_ERROR_CHECK(cuMemFree(a));
    CU_ERROR_CHECK(cuMemFree(b));
    CU_ERROR_CHECK(cuCtxPopCurrent(&contexts[0][0]));
  }

  for (int i = 0; i < count; i++) {
    for (size_t j = 0; j < 2; j++)
      CU_ERROR_CHECK(cuCtxDestroy(contexts[i][j]));
  }

  return 0;
}
