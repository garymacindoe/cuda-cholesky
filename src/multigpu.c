#include "multigpu.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

struct __cutask_st {
  CUresult (*function)(const void *);   // The function to run
  void * args;                          // Arguments for the function
  CUresult result;                      // Result of the function
  bool complete;                        // Flag set when function is finished
  pthread_mutex_t mutex;                // Mutex to protect access to result and flag
  pthread_cond_t cond;                  // Condition to wait on function completion
};

struct __cumultigpu_st {
  struct __cutask_st * tasks;   // Queue of tasks
  size_t head, tail, capacity;  // Head, tail and capacity of queue
  pthread_t * threads;          // Background threads
  int n;                        // Number of background threads
  pthread_mutex_t mutex;        // Mutex to protect access to queue
  pthread_cond_t cond;          // Condition to wait on non-empty queue
};

#define PTHREAD_ERROR_CHECK(call) \
  do { \
    int error = call; \
    if (error != 0) { \
      errno = error; \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, strerror); \
      return CUDA_ERROR_OPERATING_SYSTEM; \
    } \
  } while (false)

/**
 * Thread cleanup function.
 */
void destroy_context(void * args) {
  CUcontext ctx = (CUcontext)args;
  CU_ERROR_CHECK_VOID(cuCtxDestroy(ctx));
}

/**
 * Background thread function.
 */
static void * thread_main(void * args) {
  // Get a pointer to the multiGPU context
  CUmultiGPU mGPU = (CUmultiGPU)args;

  // Create a context on the device using the context flags
  CUcontext ctx;
  PTHREAD_CU_ERROR_CHECK(cuCtxCreate(&ctx, mGPU->flags, mGPU->device));

  // Register a thread exit handler to destroy the context
  PTHREAD_ERROR_CHECK_EXIT(pthread_cleanup_push(destroy_context, ctx));

  // Enter main loop (pthread_cancel will stop thread execution)
  while (true) {
    // Lock the mutex for the task queue
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_lock(&mGPU->mutex));

    // Wait until there is a task in the queue
    while (mGPU->head == mGPU->tail)
      PTHREAD_ERROR_CHECK_EXIT(pthread_cond_wait(&mGPU->cond, &mGPU->mutex));

    // Remove the task from the queue
    struct __cutask_st task = mGPU->tasks[mGPU->head++];
    if (mGPU->head == mGPU->capacity)
      mGPU->head = 0;

    // Unlock the mutex for the task queue
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_unlock(&mGPU->mutex));

    // Lock the mutex for the task
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_lock(&task.mutex));

    // Run the task function using the arguments and assign the result
    task.result = task.function(task.args);

    // Set the task as completed
    task.complete = true;

    // Unlock the task mutex
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_unlock(&task.mutex));

    // Signal to waiting threads that the task has now completed
    PTHREAD_ERROR_CHECK_EXIT(pthread_cond_signal(&task.cond));
  }

  // Exit the thread (never reached)
  pthread_exit(NULL);
}

/**
 * Creates a multiGPU context with a number of background threads each with a
 * context for a device and with a shared queue of tasks to execute.
 *
 * @param multiGPU  the handle to the created context is returned through this
 *                  pointer.
 * @param flags     flags for the context created on each device.
 * @param devices   the devices to create the contexts on.
 * @param n         the number of devices.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
 *         CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM,
 *         CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
 */
CUresult cuMultiGPUCreate(CUmultiGPU * multiGPU, unsigned int flags, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  // Allocate space for a new multigpu context on the heap
  CUmultiGPU mGPU;
  if ((mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Set up the queue
  mGPU->head = 0;
  mGPU->tail = 0;
  mGPU->capacity = 16;

  // Allocate space for the task queue
  if ((mGPU->tasks = malloc(mGPU->capacity * sizeof(struct __cutask_st))) == NULL) {
    free(mGPU);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Allocate space for the threads
  if ((mGPU->threads = malloc((size_t)n * sizeof(pthread_t))) == NULL) {
    free(mGPU->tasks);
    free(mGPU);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Set up the mutex and condition variable
  mGPU->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  mGPU->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // From this point on the pool is functional even if no threads start so
  // assign the result so it can be destroyed later
  *multiGPU = mGPU;

  // Start the background threads
  for (mGPU->n = 0; mGPU->n < n; mGPU->n++)
    PTHREAD_ERROR_CHECK(pthread_create(&mGPU->threads[mGPU->n], NULL, thread_main, mGPU));

  return 0;
}

/**
 * Destroys a multiGPU context.  Any tasks currently scheduled will not be run
 * and any currently running will be cancelled.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU multiGPU) {
  // Cancel each thread and wait for it to finish
  for (int i = 0; i < pool->n; i++) {
    PTHREAD_ERROR_CHECK(pthread_cancel(pool->threads[i]));
    PTHREAD_ERROR_CHECK(pthread_join(pool->threads[i], NULL));
  }

  // Destroy the rest of the tasks in the queue
  while (pool->head != pool->tail) {
    struct __cutask_st task = pool->tasks[pool->head++];
    if (pool->head == pool->capacity)
      pool->head = 0;
    free(task.args);
  }

  // Free the queue
  free(pool->tasks);
  free(pool);

  return 0;
}

/**
 * Schedules a task to run on a GPU.
 *
 * @param task      a handle to the background task is returned through this
 *                  pointer.
 * @param multiGPU  the multiGPU context to use.
 * @param function  the function to run.
 * @param args      the parameters for the function (may be NULL).
 * @param n         the size of the parameter object (must be zero if args is NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuTaskSchedule(CUtask * task, CUmultiGPU multiGPU, CUresult (*function)(const void *),
                        const void * args, size_t n) {
  // Function cannot be NULL or the background thread will SIGSEGV
  // This function will SIGSEGV if args is NULL and the size is > 0
  if (function == NULL || (args == NULL && size > 0))
    return CUDA_ERROR_INVALID_VALUE;

  // Create a new task structure
  struct __task_st * t;
  if ((t = malloc(sizeof(struct __task_st))) == NULL)
    return (errno = ENOMEM);

  // Allocate space on the heap for function arguments (arguments on stack are
  // not shared between threads)
  if ((t->args = malloc(size)) == NULL) {
    free(t);
    return (errno = ENOMEM);
  }

  // Assign task fields
  t->function = function;
  t->args = memcpy(t->args, args, size);
  t->complete = false;
  t->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  t->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Lock the mutex for the queue
  int error;
  if ((error = pthread_mutex_lock(&pool->mutex)) != 0) {
    free(t->args);
    free(t);
    return (errno = error);
  }

  // Add the task to the queue
  pool->tasks[pool->tail++] = *t;

  // Assign the result handle
  *task = t;

  // Wrap the tail around the queue
  if (pool->tail == pool->capacity)
    pool->tail = 0;

  // Increase the queue capacity if needed
  if (pool->tail == pool->head) {
    // Double the capacity
    size_t capacity = pool->capacity * 2;
    // Allocate a new array for the queue
    struct __task_st * ptr = malloc(capacity * sizeof(struct __task_st));
    if (ptr == NULL)
      return (errno = ENOMEM);

    // Copy the tasks into the new array
    if (pool->head < pool->tail)
      memcpy(ptr, &pool->tasks[pool->head], (pool->tail - pool->head) * sizeof(struct __task_st));
    else {
      memcpy(ptr, &pool->tasks[pool->head], (pool->capacity - pool->head) * sizeof(struct __task_st));
      memcpy(&ptr[pool->capacity - pool->head], pool->tasks, pool->tail * sizeof(struct __task_st));
    }

    // Update the pool with the new queue
    pool->tasks = ptr;
    pool->head = 0;
    pool->tail = pool->capacity;
    pool->capacity = capacity;
  }

  // Unlock the queue mutex
  if ((error = pthread_mutex_unlock(&pool->mutex)) != 0)
    return (errno = error);

  // Signal to background threads that there are now tasks available
  if ((error = pthread_cond_signal(&pool->cond)) != 0)
    return (errno = error);

  return 0;
}

/**
 * Destroys the background task, blocking until it has been completed.
 *
 * @param task  the task.
 * @param res   the result of the backgound task is returned through this pointer.
 * @return EINVAL, EAGAIN, EDEADLK.
 */
int task_destroy(task_t task, int * res) {
  int error;

  // Lock the mutex for the task
  if ((error = pthread_mutex_lock(&task->mutex)) != 0)
    return (errno = error);

  // Wait until the task has been completed
  while (!task->complete) {
    if ((error = pthread_cond_wait(&task->cond, &task->mutex)) != 0)
      return (errno = error);
  }

  // Copy the result
  *res = task->result;

  // Unlock the task mutex
  if ((error = pthread_mutex_unlock(&task->mutex)) != 0)
    return (errno = error);

  // Free the task and arguments
  free(task->args);
  free(task);

  return 0;
}
