#ifndef TASK_H
#define TASK_H

#include <stddef.h>

/**
 * Task type.
 */
typedef struct __task_st * task_t;

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
int task_create(task_t *, int (*)(const void *), const void *, size_t);

/**
 * Destroys the the task.  If the task has not yet completed this will block
 * until it has.
 *
 * @param task    the task to destroy.
 * @param result  the result returned by the background task (may be NULL).
 * @return zero on success.
 */
int task_destroy(task_t, int *);

/**
 * Executes the task on the calling thread.
 *
 * @param task  the task to execute.
 * @return zero on success.
 */
int task_execute(task_t);

#endif
