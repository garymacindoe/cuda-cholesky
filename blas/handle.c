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

  // Stick the LAPACK kernels in here for now too
  handle->spotrf = NULL;
  handle->strtri = NULL;
  handle->slauum = NULL;

  handle->cpotrf = NULL;
  handle->ctrtri = NULL;
  handle->clauum = NULL;

  handle->dpotrf = NULL;
  handle->dtrtri = NULL;
  handle->dlauum = NULL;

  handle->zpotrf = NULL;
  handle->ztrtri = NULL;
  handle->zlauum = NULL;

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

  if (handle->spotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->spotrf));
  if (handle->strtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strtri));
  if (handle->slauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->slauum));

  if (handle->cpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cpotrf));
  if (handle->ctrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrtri));
  if (handle->clauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->clauum));

  if (handle->dpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dpotrf));
  if (handle->dtrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrtri));
  if (handle->dlauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dlauum));

  if (handle->zpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zpotrf));
  if (handle->ztrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrtri));
  if (handle->zlauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zlauum));

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
