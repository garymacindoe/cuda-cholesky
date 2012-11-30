#include "multigpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "error.h"

#define CU_ERROR_HANDLER(call, result) \
  do { \
    if (cuErrorHandler != NULL) \
      cuErrorHandler(call, __func__, __FILE__, __LINE__, result); \
  } while (false)

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

/**
 * Thread cleanup function.
 */
void destroy_context(void * args) {
  CUcontext ctx = (CUcontext)args;
  CU_ERROR_CHECK_VOID(cuCtxDestroy(ctx));
}

/**
 * Thread arguments.
 */
struct __thread_args {
  CUmultiGPU mGPU;      // Task queue
  unsigned int flags;   // Context flags
  CUdevice device;      // Device
};

/**
 * Background thread function.
 */
static void * thread_main(void * args) {
  // Unpack the arguments
  struct __thread_args * thread_args = (struct __thread_args *)args;
  CUmultiGPU mGPU = thread_args->mGPU;
  unsigned int flags = thread_args->flags;
  CUdevice device = thread_args->device;
  free(args);

  CUresult result;
  int error;

  // Create a context on the device using the context flags
  CUcontext ctx;
  if ((result = cuCtxCreate(&ctx, flags, device)) != CUDA_SUCCESS) {
    CU_ERROR_HANDLER("cuCtxCreate", result);
    pthread_exit((void *)result);
  }

  // Register a thread exit handler to destroy the context
  pthread_cleanup_push(destroy_context, ctx);

  // Enter main loop (pthread_cancel will stop thread execution)
  while (true) {
    // Lock the mutex for the task queue
    if ((error = pthread_mutex_lock(&mGPU->mutex)) != 0)
      fprintf(stderr, "Unable to lock queue mutex: %s\n", strerror(error));

    // Wait until there is a task in the queue
    while (mGPU->head == mGPU->tail) {
      if ((error = pthread_cond_wait(&mGPU->cond, &mGPU->mutex)) != 0)
        fprintf(stderr, "Unable to wait on queue: %s\n", strerror(error));
    }

    // Remove the task from the queue
    struct __cutask_st task = mGPU->tasks[mGPU->head++];
    if (mGPU->head == mGPU->capacity)
      mGPU->head = 0;

    // Unlock the mutex for the task queue
    if ((error = pthread_mutex_unlock(&mGPU->mutex)) != 0)
      fprintf(stderr, "Unable to unlock queue mutex: %s\n", strerror(error));

    // Lock the mutex for the task
    if ((error = pthread_mutex_lock(&task.mutex)) != 0)
      fprintf(stderr, "Unable to lock task mutex: %s\n", strerror(error));

    // Run the task function using the arguments and assign the result
    task.result = task.function(task.args);

    // Set the task as completed
    task.complete = true;

    // Unlock the task mutex
    if ((error = pthread_mutex_unlock(&task.mutex)) != 0)
      fprintf(stderr, "Unable to unlock task mutex: %s\n", strerror(error));

    // Signal to waiting threads that the task has now completed
    if ((error = pthread_cond_signal(&task.cond)) != 0)
      fprintf(stderr, "Unable to signal threads waiting on task result: %s\n", strerror(error));
  }

  // Pop the cleanup handler and execute it
  pthread_cleanup_pop(1);

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
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM,
 *         CUDA_ERROR_OUT_OF_MEMORY
 */
CUresult cuMultiGPUCreate(CUmultiGPU * multiGPU, unsigned int flags,
                          CUdevice * devices, int n) {
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
  for (mGPU->n = 0; mGPU->n < n; mGPU->n++) {
    // Create thread arguments
    struct __thread_args * args = malloc(sizeof(struct __thread_args));
    if (args == NULL)
      return CUDA_ERROR_OUT_OF_MEMORY;

    // Set up arguments
    args->mGPU = mGPU;
    args->flags = flags;
    args->device = devices[mGPU->n];

    // Launch thread
    if (pthread_create(&mGPU->threads[mGPU->n], NULL, thread_main, args) != 0) {
      free(args);
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys a multiGPU context.  Any tasks currently scheduled will not be run
 * and any currently running will be cancelled.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU multiGPU) {
  // Cancel each thread and wait for it to finish
  for (int i = 0; i < multiGPU->n; i++) {
    if (pthread_cancel(multiGPU->threads[i]) != 0 ||
        pthread_join(multiGPU->threads[i], NULL) != 0)
      return CUDA_ERROR_OPERATING_SYSTEM;
  }

  // Destroy the rest of the tasks in the queue
  while (multiGPU->head != multiGPU->tail) {
    struct __cutask_st task = multiGPU->tasks[multiGPU->head++];
    if (multiGPU->head == multiGPU->capacity)
      multiGPU->head = 0;
    free(task.args);
  }

  // Free the queue
  free(multiGPU->tasks);
  free(multiGPU);

  return CUDA_SUCCESS;
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
CUresult cuTaskSchedule(CUtask * task, CUmultiGPU multiGPU,
                        CUresult (*function)(const void *), const void * args,
                        size_t n) {
  // Function cannot be NULL or the background thread will SIGSEGV
  // This function will SIGSEGV if args is NULL and the size is > 0
  if (function == NULL || (args == NULL && n > 0))
    return CUDA_ERROR_INVALID_VALUE;

  // Create a new task structure
  struct __cutask_st * t;
  if ((t = malloc(sizeof(struct __cutask_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Allocate space on the heap for function arguments (arguments on stack are
  // not shared between threads)
  if ((t->args = malloc(n)) == NULL) {
    free(t);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Assign task fields
  t->function = function;
  t->args = memcpy(t->args, args, n);
  t->complete = false;
  t->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  t->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  // Lock the mutex for the queue
  if (pthread_mutex_lock(&multiGPU->mutex) != 0) {
    free(t->args);
    free(t);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }

  // Add the task to the queue
  multiGPU->tasks[multiGPU->tail++] = *t;

  // Assign the result handle
  *task = t;

  // Wrap the tail around the queue
  if (multiGPU->tail == multiGPU->capacity)
    multiGPU->tail = 0;

  // Increase the queue capacity if needed
  if (multiGPU->tail == multiGPU->head) {
    // Double the capacity
    size_t capacity = multiGPU->capacity * 2;
    // Allocate a new array for the queue
    struct __cutask_st * ptr = malloc(capacity * sizeof(struct __cutask_st));
    if (ptr == NULL) {
      pthread_mutex_unlock(&multiGPU->mutex);
      free(t->args);
      free(t);
      return CUDA_ERROR_OPERATING_SYSTEM;
    }

    // Copy the tasks into the new array
    if (multiGPU->head < multiGPU->tail)
      memcpy(ptr, &multiGPU->tasks[multiGPU->head], (multiGPU->tail - multiGPU->head) * sizeof(struct __cutask_st));
    else {
      memcpy(ptr, &multiGPU->tasks[multiGPU->head], (multiGPU->capacity - multiGPU->head) * sizeof(struct __cutask_st));
      memcpy(&ptr[multiGPU->capacity - multiGPU->head], multiGPU->tasks, multiGPU->tail * sizeof(struct __cutask_st));
    }

    // Update the pool with the new queue
    multiGPU->tasks = ptr;
    multiGPU->head = 0;
    multiGPU->tail = multiGPU->capacity;
    multiGPU->capacity = capacity;
  }

  // Unlock the queue mutex
  if (pthread_mutex_unlock(&multiGPU->mutex) != 0)
    return CUDA_ERROR_OPERATING_SYSTEM;

  // Signal to background threads that there are now tasks available
  if (pthread_cond_signal(&multiGPU->cond) != 0)
    return CUDA_ERROR_OPERATING_SYSTEM;

  return CUDA_SUCCESS;
}

/**
 * Destroys the the background task.  If the task has not yet completed this
 * will block until it has.
 *
 * @param task    the background task to destroy.
 * @param result  the result returned by the background task.
 * @return CUDA_SUCCESS, CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuTaskDestroy(CUtask task, CUresult * res) {
  // Lock the mutex for the task
  if (pthread_mutex_lock(&task->mutex) != 0)
    return CUDA_ERROR_OPERATING_SYSTEM;

  // Wait until the task has been completed
  while (!task->complete) {
    if (pthread_cond_wait(&task->cond, &task->mutex) != 0) {
      pthread_mutex_unlock(&task->mutex);
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
  }

  // Copy the result
  *res = task->result;

  // Unlock the task mutex
  if (pthread_mutex_unlock(&task->mutex) != 0)
    return CUDA_ERROR_OPERATING_SYSTEM;

  // Free the task and arguments
  free(task->args);
  free(task);

  return CUDA_SUCCESS;
}
