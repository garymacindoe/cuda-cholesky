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

  handle->sgemm2 = NULL;
  handle->ssyrk = NULL;
  handle->strmm2 = NULL;
  handle->strsm = NULL;

  handle->cgemm2 = NULL;
  handle->cherk = NULL;
  handle->ctrmm2 = NULL;
  handle->ctrsm = NULL;

  handle->dgemm2 = NULL;
  handle->dsyrk = NULL;
  handle->dtrmm2 = NULL;
  handle->dtrsm = NULL;

  handle->zgemm2 = NULL;
  handle->zherk = NULL;
  handle->ztrmm2 = NULL;
  handle->ztrsm = NULL;

  return CUDA_SUCCESS;
}

static inline CUresult cublashandle_cleanup(struct __cublashandle_st * handle) {
  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->sgemm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->sgemm2));
  if (handle->ssyrk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ssyrk));
  if (handle->strmm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strmm2));
  if (handle->strsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strsm));

  if (handle->cgemm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cgemm2));
  if (handle->cherk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cherk));
  if (handle->ctrmm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrmm2));
  if (handle->ctrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrsm));

  if (handle->dgemm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dgemm2));
  if (handle->dsyrk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dsyrk));
  if (handle->dtrmm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrmm2));
  if (handle->dtrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrsm));

  if (handle->zgemm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zgemm2));
  if (handle->zherk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zherk));
  if (handle->ztrmm2 != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrmm2));
  if (handle->ztrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrsm));

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
