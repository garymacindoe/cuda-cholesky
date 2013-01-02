#include "util/task.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>

/**
 * Task structure.
 */
struct __task_st {
  int (*function)(const void *);       /** The function to run                */
  void * args;                         /** Arguments for the function         */
  int result;                          /** Result of the function             */
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
 * @return zero on success,
 *         EINVAL if <b>function</b> is NULL or <b>args</b> is NULL and
 *         <b>size</b> is greater than 0,
 *         ENOMEM if there is not enough memory to create another task.
 */
int task_create(task_t * task, int (*function)(const void *), const void * args,
                size_t size) {
  if (function == NULL || (args == NULL && size > 0))
    return EINVAL;

  if (((*task) = malloc(sizeof(struct __task_st))) == NULL)
    return ENOMEM;

  (*task)->function = function;
  (*task)->complete = false;
  (*task)->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  (*task)->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  if (((*task)->args = malloc(size)) == NULL) {
    free(*task);
    return ENOMEM;
  }

  // Copy arguments
  (*task)->args = memcpy((*task)->args, args, size);

  return 0;
}

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return zero on success.
 */
int task_destroy(task_t task, int * result) {
  int error;

  // Lock the mutex for the task
  if ((error = pthread_mutex_lock(&task->mutex)) != 0)
    return error;

  // Wait until the task has been completed
  while (!task->complete) {
    if ((error = pthread_cond_wait(&task->cond, &task->mutex)) != 0)
      return error;
  }

  // Copy the result
  if (result != NULL)
    *result = task->result;

  // Unlock the task mutex
  if ((error = pthread_mutex_unlock(&task->mutex)) != 0)
    return error;

  // Free the task and arguments
  free(task->args);
  free(task);

  return 0;
}

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return zero on success.
 */
int task_execute(task_t task) {
  int error;

  // Lock the mutex for the task
  if ((error = pthread_mutex_lock(&task->mutex)) != 0)
    return error;

  // Run the task function using the arguments and assign the result
  task->result = task->function(task->args);

  // Set the task as completed
  task->complete = true;

  // Unlock the task mutex
  if ((error = pthread_mutex_unlock(&task->mutex)) != 0)
    return error;

  // Signal to waiting threads that the task has now completed
  if ((error = pthread_cond_signal(&task->cond)) != 0)
    return error;

  return 0;
}
