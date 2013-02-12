#include "blas.h"
#include "handle.h"
#include "error.h"

static inline CUresult cublashandle_init(struct __cublashandle_st * handle) {
  CU_ERROR_CHECK(cuCtxGetCurrent(&handle->context));
  if (handle->context == NULL) {
    CUdevice device;
    CU_ERROR_CHECK(cuDeviceGet(&device, 0));
    CU_ERROR_CHECK(cuCtxCreate(&handle->context, CU_CTX_SCHED_AUTO, device));
    handle->contextOwner = true;
  }
  else
    handle->contextOwner = false;

  handle->sgemm = NULL;
  handle->ssyrk = NULL;
  handle->strmm2 = NULL;
  handle->strsm = NULL;

  return CUDA_SUCCESS;
}

static inline CUresult cublashandle_cleanup(struct __cublashandle_st * handle) {
  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->sgemm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->sgemm));
  if (handle->ssyrk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ssyrk));
  if (handle->strmm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strmm2));
  if (handle->strsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strsm));

  if (handle->contextOwner)
    CU_ERROR_CHECK(cuCtxDestroy(handle->context));

  return CUDA_SUCCESS;
}

CUresult cuBLASCreate(CUBLAShandle * handle) {
  if ((*handle = malloc(sizeof(struct __cublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  CU_ERROR_CHECK(cublashandle_init(*handle));
  return CUDA_SUCCESS;
}

CUresult cuBLASDestroy(CUBLAShandle handle) {
  CU_ERROR_CHECK(cublashandle_cleanup(handle));
  free(handle);
  return CUDA_SUCCESS;
}

static CUresult init(const void * args) {
  CUBLAShandle handle = (CUBLAShandle)args;
  CU_ERROR_CHECK(cuBLASCreate(&handle));
  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * args) {
  CUBLAShandle handle = (CUBLAShandle)args;
  CU_ERROR_CHECK(cuBLASDestroy(handle));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBLASCreate(CUmultiGPUBLAShandle * handle, CUmultiGPU mGPU) {
  if ((*handle = malloc(sizeof(struct __cumultigpublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*handle)->mGPU = mGPU;

  int n = cuMultiGPUGetContextCount(mGPU);
  if (((*handle)->handles = malloc((size_t)n * sizeof(struct __cublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  for (int i = 0; i < n; i++) {
    CUBLAShandle h = &(*handle)->handles[i];

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &h, sizeof(CUBLAShandle)));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBLASDestroy(CUmultiGPUBLAShandle handle) {
  int n = cuMultiGPUGetContextCount(handle->mGPU);
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &handle->handles[i], sizeof(CUBLAShandle)));
    CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBLASSynchronize(CUmultiGPUBLAShandle handle) {
  CU_ERROR_CHECK(cuMultiGPUSynchronize(handle->mGPU));
  return CUDA_SUCCESS;
}
