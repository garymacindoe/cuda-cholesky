#include "multigpu.h"
#include "task.h"
#include "taskqueue.h"
#include "error.h"

#ifdef MGPU_SEQ
// Single threaded version of CUmultiGPU.

/**
 * MultiGPU context.  Single threaded version is simply an array of CUDA
 * contexts.
 */
struct __cumultigpu_st {
  CUcontext * contexts;
  int n;
};

/**
 * Creates a multiGPU context with a single CUDA context created on each of the
 * devices given.
 *
 * @param mGPU     the newly created context is returned through this pointer.
 * @param devices  the CUDA devices to use.
 * @param n        the number of CUDA devices to use.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUCreate(CUmultiGPU * mGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  if ((*mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  if (((*mGPU)->contexts = malloc((size_t)n * sizeof(CUcontext))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*mGPU)->n = n;
  for (int i = 0; i < n; i++)
    CU_ERROR_CHECK(cuCtxCreate(&(*mGPU)->contexts[i], CU_CTX_SCHED_AUTO, devices[i]));

  return CUDA_SUCCESS;
}

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++)
    CU_ERROR_CHECK(cuCtxDestroy(mGPU->contexts[i]));
  free(mGPU->contexts);
  free(mGPU);
  return CUDA_SUCCESS;
}

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU mGPU, int i, CUtask task) {
  if (i < 0 || i >= mGPU->n)
    return CUDA_ERROR_INVALID_VALUE;

  CU_ERROR_CHECK(cuCtxPushCurrent(mGPU->contexts[i]));
  CU_ERROR_CHECK(cuTaskExecute(task));
  CU_ERROR_CHECK(cuCtxPopCurrent(&mGPU->contexts[i]));

  return CUDA_SUCCESS;
}

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++) {
    CU_ERROR_CHECK(cuCtxPushCurrent(mGPU->contexts[i]));
    CU_ERROR_CHECK(cuCtxSynchronize());
    CU_ERROR_CHECK(cuCtxPopCurrent(&mGPU->contexts[i]));
  }
  return CUDA_SUCCESS;
}

#else
// Multi-threaded version of CUmultiGPU.
#include "thread.h"

/**
 * MultiGPU context.  Multithreaded version is an array of CUthreads.
 */
struct __cumultigpu_st {
  CUthread * threads;
  int n;
};

static CUresult createContext(const void * args) {
  CUdevice * device = (CUdevice *)args;
  CUcontext context;
  CU_ERROR_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_YIELD, *device));
  return CUDA_SUCCESS;
}

static CUresult destroyContext(const void * args) {
  (void)args;

  CUcontext context;
  CU_ERROR_CHECK(cuCtxGetCurrent(&context));
  CU_ERROR_CHECK(cuCtxDestroy(context));

  return CUDA_SUCCESS;
}

/**
 * Creates a multiGPU context with a single CUDA context created on each of the
 * devices given.
 *
 * @param mGPU     the newly created context is returned through this pointer.
 * @param devices  the CUDA devices to use.
 * @param n        the number of CUDA devices to use.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUCreate(CUmultiGPU * mGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  if ((*mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  if (((*mGPU)->threads = malloc((size_t)n * sizeof(CUthread))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*mGPU)->n = n;
  for (int i = 0; i < n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, createContext, &devices[i], sizeof(CUdevice)));

    CU_ERROR_CHECK(cuThreadCreate(&(*mGPU)->threads[i]));
    CU_ERROR_CHECK(cuThreadRunTask((*mGPU)->threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return (int)result;
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU mGPU) {
  for (int i = 0; i < mGPU->n; i++) {
    CUtask task;
    CU_ERROR_CHECK(cuTaskCreate(&task, destroyContext, NULL, 0));
    CU_ERROR_CHECK(cuThreadRunTask(mGPU->threads[i], task));

    CUresult result;
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;

    CU_ERROR_CHECK(cuThreadDestroy(mGPU->threads[i]));
  }
  free(mGPU->threads);
  free(mGPU);
  return CUDA_SUCCESS;
}

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU mGPU, int i, CUtask task) {
  if (i < 0 || i >= mGPU->n)
    return CUDA_ERROR_INVALID_VALUE;
  CU_ERROR_CHECK(cuThreadRunTask(mGPU->threads[i], task));
  return CUDA_SUCCESS;
}

static CUresult synchronize() {
  CU_ERROR_CHECK(cuCtxSynchronize());
  return CUDA_SUCCESS;
}

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU mGPU) {
  CUtask task;
  CUresult result;

  for (int i = 0; i < mGPU->n; i++) {
    CU_ERROR_CHECK(cuTaskCreate(&task, synchronize, NULL, 0));
    CU_ERROR_CHECK(cuMultiGPURunTask(mGPU, i, task));
    CU_ERROR_CHECK(cuTaskDestroy(task, &result));
    if (result != CUDA_SUCCESS)
      return result;
  }

  return CUDA_SUCCESS;
}

#endif

/**
 * Gets the number of contexts available in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return the number of contexts in the multiGPU context.
 */
int cuMultiGPUGetContextCount(CUmultiGPU mGPU) {
  return mGPU->n;
}
