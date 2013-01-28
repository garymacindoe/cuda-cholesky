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
  handle->strmm = NULL;
  handle->strsm = NULL;

  handle->cgemm = NULL;
  handle->cherk = NULL;
  handle->ctrmm = NULL;
  handle->ctrsm = NULL;

  handle->dgemm = NULL;
  handle->dsyrk = NULL;
  handle->dtrmm = NULL;
  handle->dtrsm = NULL;

  handle->zgemm = NULL;
  handle->zherk = NULL;
  handle->ztrmm = NULL;
  handle->ztrsm = NULL;

  return CUDA_SUCCESS;
}

static inline CUresult cublashandle_cleanup(struct __cublashandle_st * handle) {
  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->sgemm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->sgemm));
  if (handle->ssyrk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ssyrk));
  if (handle->strmm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strmm));
  if (handle->strsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strsm));

  if (handle->cgemm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cgemm));
  if (handle->cherk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cherk));
  if (handle->ctrmm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrmm));
  if (handle->ctrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrsm));

  if (handle->dgemm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dgemm));
  if (handle->dsyrk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dsyrk));
  if (handle->dtrmm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrmm));
  if (handle->dtrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrsm));

  if (handle->zgemm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zgemm));
  if (handle->zherk != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zherk));
  if (handle->ztrmm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrmm));
  if (handle->ztrsm != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrsm));

  if (handle->contextOwner)
    CU_ERROR_CHECK(cuCtxDestroy(handle->context));

  return CUDA_SUCCESS;
}

CUresult cuBlasHandleCreate(CUblashandle * handle) {
  if ((*handle = malloc(sizeof(struct __cublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  CU_ERROR_CHECK(cublashandle_init(*handle));
  return CUDA_SUCCESS;
}

CUresult cuBlasHandleDestroy(CUblashandle handle) {
  CU_ERROR_CHECK(cublashandle_cleanup(handle));
  free(handle);
  return CUDA_SUCCESS;
}

static CUresult init(const void * args) {
  CUblashandle handle = (CUblashandle)args;
  CU_ERROR_CHECK(cuBlasHandleCreate(&handle));
  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * args) {
  CUblashandle handle = (CUblashandle)args;
  CU_ERROR_CHECK(cuBlasHandleDestroy(handle));
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBlasCreate(CUmultiGPUBlasHandle * handle, CUmultiGPU mGPU) {
  if ((*handle = malloc(sizeof(struct __cumultigpublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*handle)->mGPU = mGPU;

  int n = cuMultiGPUGetContextCount(mGPU);
  if (((*handle)->handles = malloc((size_t)n * sizeof(struct __cublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  for (int i = 0; i < n; i++) {
    CUblashandle h = &(*handle)->handles[i];

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &h, sizeof(CUblashandle)));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBlasDestroy(CUmultiGPUBlasHandle handle) {
  int n = cuMultiGPUGetContextCount(handle->mGPU);
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &handle->handles[i], sizeof(CUblashandle)));
    CU_ERROR_CHECK(cuMultiGPURunTask(handle->mGPU, i, task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBlasSynchronize(CUmultiGPUBlasHandle handle) {
  CU_ERROR_CHECK(cuMultiGPUSynchronize(handle->mGPU));
  return CUDA_SUCCESS;
}
