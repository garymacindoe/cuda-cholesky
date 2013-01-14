#ifndef CUMULTIGPU_H
#define CUMULTIGPU_H

#include <stddef.h>
#include <cuda.h>

typedef struct __cutask_st * CUtask;

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
CUresult cuTaskCreate(CUtask *, CUresult (*)(const void *), const void *, size_t);

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask, CUresult *);

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskExecute(CUtask);

/**
 * MultiGPU context.  May be single threaded or multi-threaded.
 */
typedef struct __cumultigpu_st * CUmultiGPU;

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
CUresult cuMultiGPUCreate(CUmultiGPU *, CUdevice *, int);

/**
 * Destroys a multiGPU context.
 *
 * @param mGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPUDestroy(CUmultiGPU);

/**
 * Gets the number of contexts available in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return the number of contexts in the multiGPU context.
 */
int cuMultiGPUGetContextCount(CUmultiGPU);

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU, int, CUtask);

/**
 * Synchronises all contexts in the multiGPU context.
 *
 * @param mGPU  the multiGPU context.
 * @return any errors.
 */
CUresult cuMultiGPUSynchronize(CUmultiGPU);

#endif
