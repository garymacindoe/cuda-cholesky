#include "cumultigpu.h"
#include "error.h"

// Single threaded versions of CUtask and CUmultiGPU.

/**
 * Task structure.
 */
struct __cutask_st {
  CUresult (*function)(const void *);  /** The function to run                */
  void * args;                         /** Arguments for the function         */
  CUresult result;                     /** Result of the function             */
};

/**
 * Creates a task.
 *
 * @param task      the newly created task is returned through this pointer.
 * @param function  the function to execute.
 * @param args      arguments for the function.
 * @param size      the size of the arguments.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_INVALID_VALUE if <b>function</b> is NULL or <b>args</b> is
 *         NULL and <b>size</b> is greater than 0,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory to create
 *         another task.
 */
CUresult cuTaskCreate(CUtask * task, CUresult (*function)(const void *),
                      const void * args, size_t size) {
  if (function == NULL || (args == NULL && size > 0))
    return CUDA_ERROR_INVALID_VALUE;

  if (((*task) = malloc(sizeof(struct __cutask_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  (*task)->function = function;

  // Allocate space on heap for arguments so that they can be accessed by other
  // threads
  if (((*task)->args = malloc(size)) == NULL) {
    free(*task);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Copy arguments onto heap
  (*task)->args = memcpy((*task)->args, args, size);

  return CUDA_SUCCESS;
}

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask task, CUresult * result) {
  // Copy the result
  if (result != NULL)
    *result = task->result;

  // Free the task and arguments
  free(task->args);
  free(task);

  return CUDA_SUCCESS;
}

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskExecute(CUtask task) {
  // Run the task function using the arguments and assign the result
  task->result = task->function(task->args);
  return CUDA_SUCCESS;
}

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
  for (int i = 0; i < n; i++) {
    CU_ERROR_CHECK(cuCtxCreate(&(*mGPU)->contexts[i], CU_CTX_SCHED_AUTO, devices[i]));
    CU_ERROR_CHECK(cuCtxPopCurrent(&(*mGPU)->contexts[i]));
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

/**
 * Gets the number of contexts available in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return the number of contexts in the multiGPU context.
 */
int cuMultiGPUGetContextCount(CUmultiGPU mGPU) {
  return mGPU->n;
}
