#include "lapack.h"
#include "handle.h"
#include "error.h"

static inline CUresult culapackhandle_init(struct __culapackhandle_st * handle) {
  CU_ERROR_CHECK(cuBLASCreate(&handle->blas_handle));
  CU_ERROR_CHECK(cuCtxGetCurrent(&handle->context));

  handle->spotrf = NULL;
  handle->strtri = NULL;
  handle->slauum = NULL;
  handle->slogdet = NULL;

  handle->cpotrf = NULL;
  handle->ctrtri = NULL;
  handle->clauum = NULL;
  handle->clogdet = NULL;

  handle->dpotrf = NULL;
  handle->dtrtri = NULL;
  handle->dlauum = NULL;
  handle->dlogdet = NULL;

  handle->zpotrf = NULL;
  handle->ztrtri = NULL;
  handle->zlauum = NULL;
  handle->zlogdet = NULL;

  return CUDA_SUCCESS;
}

static inline CUresult culapackhandle_cleanup(struct __culapackhandle_st * handle) {
  CU_ERROR_CHECK(cuCtxPushCurrent(handle->context));

  if (handle->spotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->spotrf));
  if (handle->strtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->strtri));
  if (handle->slauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->slauum));
  if (handle->slogdet != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->slogdet));

  if (handle->cpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->cpotrf));
  if (handle->ctrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ctrtri));
  if (handle->clauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->clauum));
  if (handle->clogdet != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->clogdet));

  if (handle->dpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dpotrf));
  if (handle->dtrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dtrtri));
  if (handle->dlauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dlauum));
  if (handle->dlogdet != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->dlogdet));

  if (handle->zpotrf != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zpotrf));
  if (handle->ztrtri != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->ztrtri));
  if (handle->zlauum != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zlauum));
  if (handle->zlogdet != NULL)
    CU_ERROR_CHECK(cuModuleUnload(handle->zlogdet));

  CU_ERROR_CHECK(cuCtxPopCurrent(&handle->context));
  CU_ERROR_CHECK(cuBLASDestroy(handle->blas_handle));

  return CUDA_SUCCESS;
}

CUresult cuLAPACKCreate(CULAPACKhandle * handle) {
  if ((*handle = malloc(sizeof(struct __culapackhandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  CU_ERROR_CHECK(culapackhandle_init(*handle));
  return CUDA_SUCCESS;
}

CUresult cuLAPACKDestroy(CULAPACKhandle handle) {
  CU_ERROR_CHECK(culapackhandle_cleanup(handle));
  free(handle);
  return CUDA_SUCCESS;
}

CUresult cuMultiGPULAPACKCreate(CUmultiGPULAPACKhandle * handle, CUmultiGPU mGPU) {
  if ((*handle = malloc(sizeof(struct __cumultigpulapackhandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  CU_ERROR_CHECK(cuMultiGPUBLASCreate(&(*handle)->blas_handle, mGPU));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPULAPACKDestroy(CUmultiGPULAPACKhandle handle) {
  CU_ERROR_CHECK(cuMultiGPUBLASDestroy(handle->blas_handle));
  free(handle);
  return CUDA_SUCCESS;
}

CUresult cuMultiGPULAPACKSynchronize(CUmultiGPULAPACKhandle handle) {
  CU_ERROR_CHECK(cuMultiGPUBLASSynchronize(handle->blas_handle));
  return CUDA_SUCCESS;
}
