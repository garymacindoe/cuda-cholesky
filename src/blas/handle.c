#include "blas.h"
#include "handle.h"
#include "error.h"

static CUresult init(const void * args) {
  struct multigpu_blas_plan * plan = *(struct multigpu_blas_plan **)args;

  // Work out the compute capability of the current device (double precision is
  // only supported on devices with compute capability 1.3 and higher)
  CUdevice device;
  CU_ERROR_CHECK(cuCtxGetDevice(&device));
  int major, minor;
  CU_ERROR_CHECK(cuDeviceComputeCapability(&major, &minor, device));
  const int cuda_arch = major * 100 + minor * 10;

  // Load the modules
  CU_ERROR_CHECK(cuModuleLoad(&plan->sgemm, "sgemm.fatbin"));
  CU_ERROR_CHECK(cuModuleLoad(&plan->cgemm, "cgemm.fatbin"));
  if (cuda_arch >= 130) {
    CU_ERROR_CHECK(cuModuleLoad(&plan->dgemm, "dgemm.fatbin"));
    CU_ERROR_CHECK(cuModuleLoad(&plan->zgemm, "zgemm.fatbin"));
  }
  else {
    plan->dgemm = 0;
    plan->zgemm = 0;
  }

  // Create two streams for concurrent copy and execute (only for page-locked
  // host to device linear (NOT pitched/2D or array) copies on some CC 1.1 and
  // later devices)
  CU_ERROR_CHECK(cuStreamCreate(&plan->copy, 0));
  CU_ERROR_CHECK(cuStreamCreate(&plan->compute, 0));

  // Allocate device memory for C (m*n)
  size_t size, mb, nb, kb;
  size = MAX(sizeof(float), sizeof(float complex));
  mb = MAX(MAX(SGEMM_N_MB, SGEMM_T_MB) * sizeof(float),
           MAX(CGEMM_N_MB, CGEMM_C_MB) * sizeof(float complex));
  nb = MAX(MAX(SGEMM_N_NB, SGEMM_T_NB), MAX(CGEMM_N_NB, CGEMM_C_NB));
  if (cuda_arch >= 130) {
    mb = MAX(mb, MAX(MAX(DGEMM_N_MB, DGEMM_T_MB) * sizeof(double),
                 MAX(ZGEMM_N_MB, MAX(ZGEMM_CN_MB, ZGEMM_CC_MB) * sizeof(double complex))));
    nb = MAX(nb, MAX(MAX(DGEMM_N_NB, DGEMM_T_NB),
                     MAX(ZGEMM_N_NB, MAX(ZGEMM_CN_NB, ZGEMM_CC_NB))));
    size = MAX(size, MAX(sizeof(double), sizeof(double complex)));
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&plan->C, &plan->ldc, mb, nb, (unsigned int)size));

  // Allocate device memory for A (m*k or k*m)
  mb = MAX(MAX(MAX(SGEMM_N_MB, SGEMM_N_KB), MAX(SGEMM_T_MB, SGEMM_T_KB)) * sizeof(float),
           MAX(MAX(CGEMM_N_MB, CGEMM_N_KB), MAX(CGEMM_C_MB, CGEMM_C_KB)) * sizeof(float complex));
  kb = MAX(MAX(MAX(SGEMM_N_KB, SGEMM_N_MB), MAX(SGEMM_T_KB, SGEMM_T_MB)),
           MAX(MAX(CGEMM_N_KB, CGEMM_N_MB), MAX(CGEMM_C_KB, CGEMM_C_MB)));
  if (cuda_arch >= 130) {
    mb = MAX(mb, MAX(MAX(MAX(DGEMM_N_MB, DGEMM_N_KB),
                         MAX(DGEMM_T_MB, DGEMM_T_KB)) * sizeof(double),
                     MAX(MAX(ZGEMM_N_MB, ZGEMM_N_KB),
                         MAX(MAX(ZGEMM_CN_MB, ZGEMM_CN_KB),
                             MAX(ZGEMM_CC_MB, ZGEMM_CC_KB))) * sizeof(double complex)));
    kb = MAX(kb, MAX(MAX(MAX(DGEMM_N_KB, DGEMM_N_MB),
                         MAX(DGEMM_T_KB, DGEMM_T_MB)),
                     MAX(MAX(ZGEMM_N_KB, ZGEMM_N_MB),
                         MAX(MAX(ZGEMM_CN_KB, ZGEMM_CN_MB),
                             MAX(ZGEMM_CC_KB, ZGEMM_CC_MB)))));
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&plan->A, &plan->lda, mb, kb * 2, (unsigned int)size));

  // Allocate device memory for B (k*n or n*k)
  kb = MAX(MAX(MAX(SGEMM_N_KB, SGEMM_N_NB), MAX(SGEMM_T_KB, SGEMM_T_NB)) * sizeof(float),
           MAX(MAX(CGEMM_N_KB, CGEMM_N_NB), MAX(CGEMM_C_KB, CGEMM_C_NB)) * sizeof(float complex));
  nb = MAX(MAX(MAX(SGEMM_N_NB, SGEMM_N_KB), MAX(SGEMM_T_NB, SGEMM_T_KB)),
           MAX(MAX(CGEMM_N_NB, CGEMM_N_KB), MAX(CGEMM_C_NB, CGEMM_C_KB)));
  if (cuda_arch >= 130) {
    kb = MAX(kb, MAX(MAX(MAX(DGEMM_N_KB, DGEMM_N_NB),
                         MAX(DGEMM_T_KB, DGEMM_T_NB)) * sizeof(double),
                     MAX(MAX(ZGEMM_N_KB, ZGEMM_N_NB),
                         MAX(MAX(ZGEMM_CN_KB, ZGEMM_CN_NB),
                             MAX(ZGEMM_CC_KB, ZGEMM_CC_NB))) * sizeof(double complex)));
    nb = MAX(nb, MAX(MAX(MAX(DGEMM_N_NB, DGEMM_N_KB),
                         MAX(DGEMM_T_NB, DGEMM_T_KB)),
                     MAX(MAX(ZGEMM_N_NB, ZGEMM_N_KB),
                         MAX(MAX(ZGEMM_CN_NB, ZGEMM_CN_KB),
                             MAX(ZGEMM_CC_NB, ZGEMM_CC_KB)))));
  }
  CU_ERROR_CHECK(cuMemAllocPitch(&plan->B, &plan->ldb, kb, nb * 2, (unsigned int)size));

  return CUDA_SUCCESS;
}

static CUresult cleanup(const void * args) {
  struct multigpu_blas_plan * plan = (struct multigpu_blas_plan *)args;

  // Free temporary memory
  CU_ERROR_CHECK(cuMemFree(plan->A));
  CU_ERROR_CHECK(cuMemFree(plan->B));
  CU_ERROR_CHECK(cuMemFree(plan->C));

  // Destroy the streams (this is asynchronous)
  CU_ERROR_CHECK(cuStreamDestroy(plan->copy));
  CU_ERROR_CHECK(cuStreamDestroy(plan->compute));

  // Unload the modules
  CU_ERROR_CHECK(cuModuleUnload(plan->sgemm));
  CU_ERROR_CHECK(cuModuleUnload(plan->dgemm));
  CU_ERROR_CHECK(cuModuleUnload(plan->cgemm));
  CU_ERROR_CHECK(cuModuleUnload(plan->zgemm));

  return CUDA_SUCCESS;
}

CUresult cuMultiGPUBlasCreate(CUmultiGPUBlasHandle * handle, CUmultiGPU mGPU) {
  if ((*handle = malloc(sizeof(struct __cumultigpublashandle_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*handle)->mGPU = mGPU;

  int n = cuMultiGPUGetContextCount(mGPU);
  if (((*handle)->plans = malloc((size_t)n * sizeof(struct multigpu_blas_plan))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  for (int i = 0; i < n; i++) {
    struct multigpu_blas_plan * plan = &(*handle)->plans[i];

    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, init, &plan, sizeof(struct multigpu_blas_plan *)));
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
    CU_ERROR_CHECK(cuTaskCreate(&task, cleanup, &handle->plans[i], sizeof(struct multigpu_blas_plan)));
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
