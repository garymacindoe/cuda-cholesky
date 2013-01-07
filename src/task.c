#include "task.h"
#include <stdlib.h>
#include <string.h>
#include "error.h"

#ifdef MGPU_SEQ
// Single threaded version of CUtask.

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

#else
// Multi-threaded version of CUtask.
#include <pthread.h>

/**
 * Task structure.
 */
struct __cutask_st {
  CUresult (*function)(const void *);  /** The function to run                */
  void * args;                         /** Arguments for the function         */
  CUresult result;                     /** Result of the function             */
  bool complete;                       /** Flag set when function is finished */
  pthread_mutex_t mutex;               /** Mutex to protect access to result and
                                           flag                               */
  pthread_cond_t cond;                 /** Condition to wait on function
                                           completion                         */
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
  (*task)->complete = false;
  (*task)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*task)->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

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
  // Lock the mutex for the task
  ERROR_CHECK(pthread_mutex_lock(&task->mutex));

  // Wait until the task has been completed
  while (!task->complete)
    ERROR_CHECK(pthread_cond_wait(&task->cond, &task->mutex));

  // Copy the result
  if (result != NULL)
    *result = task->result;

  // Unlock the task mutex
  ERROR_CHECK(pthread_mutex_unlock(&task->mutex));

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
  // Lock the mutex for the task
  ERROR_CHECK(pthread_mutex_lock(&task->mutex));

  // Run the task function using the arguments and assign the result
  task->result = task->function(task->args);

  // Set the task as completed
  task->complete = true;

  // Unlock the task mutex
  ERROR_CHECK(pthread_mutex_unlock(&task->mutex));

  // Signal to waiting threads that the task has now completed
  ERROR_CHECK(pthread_cond_signal(&task->cond));

  return CUDA_SUCCESS;
}

#endif
