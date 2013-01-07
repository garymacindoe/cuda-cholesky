#ifndef THREAD_H
#define THREAD_H

#include <cuda.h>
#include "task.h"

/**
 * Background thread type.
 */
typedef struct __cuthread_st * CUthread;

/**
 * Starts a background thread.
 *
 * @param thread  a handle to the background thread is returned through this
 *                pointer.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory,
 *         CUDA_ERROR_OPERATING_SYSTEM if the thread could not be started.
 */
CUresult cuThreadCreate(CUthread *);

/**
 * Destroys the background thread.  This function will block until all tasks are
 * completed.
 *
 * @param thread  the thread to destroy.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OPERATING_SYSTEM if the thread could not be stopped.
 */
CUresult cuThreadDestroy(CUthread);

/**
 * Schedules a task to run on a background thread.
 *
 * @param thread  the background thread to run the task on.
 * @param task    the task to run.
 * @return CUDA_SUCCESS on success,
 *         CUDA_ERROR_OUT_OF_MEMORY if there is not enough memory,
 *         CUDA_ERROR_OPERATING_SYSTEM if there was a problem communicating the
 *         task to the thread.
 */
CUresult cuThreadRunTask(CUthread, CUtask);

#endif
