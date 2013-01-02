#ifndef THREAD_H
#define THREAD_H

#include "task.h"

/**
 * Background thread type.
 */
typedef struct __thread_st * thread_t;

/**
 * Starts a background thread.
 *
 * @param thread  a handle to the background thread is returned through this
 *                pointer.
 * @return zero on success,
 *         ENOMEM if there is not enough memory.
 */
int thread_create(thread_t *);

/**
 * Destroys the background thread.  This function will block until all tasks are
 * completed.
 *
 * @param thread  the thread to destroy.
 * @return zero on success.
 */
int thread_destroy(thread_t);

/**
 * Schedules a task to run on a background thread.
 *
 * @param thread  the background thread to run the task on.
 * @param task    the task to run.
 * @return zero on success,
 *         ENOMEM if there is not enough memory.
 */
int thread_run_task(thread_t, task_t);

#endif
