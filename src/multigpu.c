#include "multigpu.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include "error.h"

#define CU_ERROR_CHECK_PTHREAD_EXIT(call) \
  do { \
    CUresult __error__; \
    if ((__error__ = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))cuGetErrorString); \
      pthread_exit((void *)(ptrdiff_t)__error__); \
    } \
  } while (false)

#define PTHREAD_ERROR_CHECK_EXIT(call) \
  do { \
    int __error__; \
    if ((__error__ = (call)) != 0) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))strerror); \
      pthread_exit((void *)(ptrdiff_t)__error__); \
    } \
  } while (false)

#define PTHREAD_ERROR_CHECK(call) \
  do { \
    int __error__; \
    if ((__error__ = (call)) != 0) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))strerror); \
      return CUDA_ERROR_OPERATING_SYSTEM; \
    } \
  } while (false)

/**
 * Rounds up to the next power of 2.
 *
 * @param n  the value to round up.
 * @return the rounded value.
 */
static inline unsigned int nextPow2(unsigned int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

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
 * Thread pool structure.
 */
struct __cumultigpu_st {
  CUtask * tasks;                 /** Queue of tasks                          */
  size_t head, tail, capacity;    /** Head, tail and capacity of queue        */
  pthread_t * threads;            /** Background threads                      */
  int n;                          /** Number of background threads            */
  pthread_mutex_t mutex;          /** Mutex to protect access to queue        */
  pthread_cond_t cond;            /** Condition to wait on non-empty queue    */
};

/**
 * Thread arguments
 */
struct thread_args {
  CUmultiGPU multiGPU;            /** MultiGPU context                        */
  CUdevice device;                /** Device to create context on             */
};

/**
 * Background thread function.
 *
 * @param args  thread arguments.
 * @return thread error status.
 */
static void * cu_thread_main(void * args) {
  struct thread_args * thread_args = (struct thread_args *)args;

  // Get a pointer to the multiGPU context
  CUmultiGPU mGPU = thread_args->multiGPU;
  CUdevice device = thread_args->device;
  free(args);

  // Create a GPU context on the specified device
  CUcontext context;
  CU_ERROR_CHECK_PTHREAD_EXIT(cuCtxCreate(&context, CU_CTX_SCHED_YIELD, device));

  // Enter main loop
  while (true) {
    // Lock the mutex for the task queue
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_lock(&mGPU->mutex));

    // Wait until there is a task in the queue
    while (mGPU->head == mGPU->tail)
      PTHREAD_ERROR_CHECK_EXIT(pthread_cond_wait(&mGPU->cond, &mGPU->mutex));

    // Remove the task from the queue
    CUtask task = mGPU->tasks[mGPU->head++];
    if (mGPU->head == mGPU->capacity)
      mGPU->head = 0;

    // Unlock the mutex for the task queue
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_unlock(&mGPU->mutex));

    // If the task function is NULL then that is the signal to break from the
    // main loop
    if (task->function == NULL) {
      free(task);
      break;
    }

    // Lock the mutex for the task
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_lock(&task->mutex));

    // Run the task function using the arguments and assign the result
    task->result = task->function(task->args);

    // Set the task as completed
    task->complete = true;

    // Unlock the task mutex
    PTHREAD_ERROR_CHECK_EXIT(pthread_mutex_unlock(&task->mutex));

    // Signal to waiting threads that the task has now completed
    PTHREAD_ERROR_CHECK_EXIT(pthread_cond_broadcast(&task->cond));
  }

  // Destroy the context
  CU_ERROR_CHECK_PTHREAD_EXIT(cuCtxDestroy(context));

  // Exit the thread
  pthread_exit(NULL);
}

/**
 * Creates a multiGPU context with a single GPU context on each device given.
 * Each context is owned by a background thread and tasks are sent to them
 * asynchronously via a shared queue.
 *
 * @param multiGPU  the handle to the created context is returned through this
 *                  pointer.
 * @param devices   devices to create contexts on.
 * @param n         the number of devices.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_DEVICE,
 *         CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OPERATING_SYSTEM,
 *         CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN
 */
CUresult cuMultiGPUCreate(CUmultiGPU * multiGPU, CUdevice * devices, int n) {
  if (n <= 0)
    return CUDA_ERROR_INVALID_VALUE;

  // Allocate space for a new thread pool on the heap
  CUmultiGPU mGPU;
  if ((mGPU = malloc(sizeof(struct __cumultigpu_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  // Set up the queue
  mGPU->head = 0;
  mGPU->tail = 0;
  // Capacity must be > n to have enough space to add cancellation tasks
  // without reallocating
  mGPU->capacity = nextPow2((unsigned int)n + 1);

  // Allocate space for the task queue
  if ((mGPU->tasks = malloc(mGPU->capacity * sizeof(CUtask))) == NULL) {
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
    struct thread_args * args;
    if ((args = malloc(sizeof(struct thread_args))) == NULL)
      return CUDA_ERROR_OUT_OF_MEMORY;

    args->multiGPU = mGPU;
    args->device = devices[mGPU->n];

    PTHREAD_ERROR_CHECK(pthread_create(&mGPU->threads[mGPU->n], NULL,
                                       cu_thread_main, args));
  }

  return CUDA_SUCCESS;
}

/**
 * Destroys the multiGPU context.  This will block until all currently executing
 * tasks have completed.  Queued tasks will be destroyed.
 *
 * @param multiGPU  the context to destroy.
 * @return CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
 *         CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuMultiGPUDestroy(CUmultiGPU multiGPU) {
  // Lock the queue
  PTHREAD_ERROR_CHECK(pthread_mutex_lock(&multiGPU->mutex));

  // Empty the queue of tasks
  while (multiGPU->head != multiGPU->tail) {
    CUtask task = multiGPU->tasks[multiGPU->head++];

    if (multiGPU->head == multiGPU->capacity)
      multiGPU->head = 0;

    free(task->args);
    free(task);
  }

  // Place one exit task for each thread on the queue
  for (int i = 0; i < multiGPU->n; i++) {
    CUtask task;
    if ((task = malloc(sizeof(struct __cutask_st))) == NULL) {
      pthread_mutex_unlock(&multiGPU->mutex);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    task->function = NULL;
    task->args = NULL;
    task->complete = false;
    task->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    task->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

    multiGPU->tasks[multiGPU->tail++] = task;

    if (multiGPU->tail == multiGPU->capacity)
      multiGPU->tail = 0;
  }

  // Unlock the queue
  PTHREAD_ERROR_CHECK(pthread_mutex_unlock(&multiGPU->mutex));

  // Signal to the threads that there are tasks waiting
  PTHREAD_ERROR_CHECK(pthread_cond_broadcast(&multiGPU->cond));

  // Wait for each thread to exit
  for (int i = 0; i < multiGPU->n; i++)
    PTHREAD_ERROR_CHECK(pthread_join(multiGPU->threads[i], NULL));

  // Free the queue
  free(multiGPU->tasks);
  free(multiGPU);

  return CUDA_SUCCESS;
}

/**
 * Schedules a task to run on a GPU managed by a background thread.
 *
 * @param task      the newly created task is returned through this handle.
 * @param multiGPU  the multiGPU context to use.
 * @param function  the function to run.
 * @param args      the parameters for the function (may be NULL).
 * @param n         the size of the parameter object (must be zero if args is
 *                  NULL).
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM
 */
CUresult cuTaskSchedule(CUtask * task, CUmultiGPU multiGPU,
                        CUresult (*function)(const void *), const void * args,
                        size_t n) {
  // Function cannot be NULL as that is the signal for the background thread to
  // exit
  // This function will SIGSEGV if args is NULL and n is > 0
  if (function == NULL || (args == NULL && n > 0))
    return CUDA_ERROR_INVALID_VALUE;

  CUtask t;
  if ((t = malloc(sizeof(struct __cutask_st))) == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;

  t->function = function;
  t->complete = false;
  t->mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
  t->cond = (pthread_cond_t)PTHREAD_COND_INITIALIZER;

  if ((t->args = malloc(n)) == NULL) {
    free(t);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Copy arguments
  t->args = memcpy(t->args, args, n);

  // Lock the mutex for the queue
  PTHREAD_ERROR_CHECK(pthread_mutex_lock(&multiGPU->mutex));

  // Add the task to the queue
  multiGPU->tasks[multiGPU->tail++] = t;

  // Wrap the tail around the queue
  if (multiGPU->tail == multiGPU->capacity)
    multiGPU->tail = 0;

  // Assign the task handle
  *task = t;

  // Increase the queue capacity if needed
  if (multiGPU->tail == multiGPU->head) {
    // Double the capacity
    size_t capacity = multiGPU->capacity * 2;
    // Allocate a new array for the queue
    CUtask * ptr = malloc(capacity * sizeof(CUtask));
    if (ptr == NULL) {
      pthread_mutex_unlock(&multiGPU->mutex);
      pthread_cond_broadcast(&multiGPU->cond);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Copy the tasks into the new array
    if (multiGPU->head < multiGPU->tail)
      memcpy(ptr, &multiGPU->tasks[multiGPU->head],
             (multiGPU->tail - multiGPU->head) * sizeof(CUtask));
    else {
      memcpy(ptr, &multiGPU->tasks[multiGPU->head],
             (multiGPU->capacity - multiGPU->head) * sizeof(CUtask));
      memcpy(&ptr[multiGPU->capacity - multiGPU->head],
             multiGPU->tasks, multiGPU->tail * sizeof(CUtask));
    }

    // Update the pool with the new queue
    multiGPU->tasks = ptr;
    multiGPU->head = 0;
    multiGPU->tail = multiGPU->capacity;
    multiGPU->capacity = capacity;
  }

  // Unlock the queue mutex
  PTHREAD_ERROR_CHECK(pthread_mutex_unlock(&multiGPU->mutex));

  // Signal to background threads that there are now tasks available
  PTHREAD_ERROR_CHECK(pthread_cond_broadcast(&multiGPU->cond));

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
CUresult cuTaskDestroy(CUtask task, CUresult * result) {
  // Lock the mutex for the task
  PTHREAD_ERROR_CHECK(pthread_mutex_lock(&task->mutex));

  // Wait until the task has been completed
  while (!task->complete)
    PTHREAD_ERROR_CHECK(pthread_cond_wait(&task->cond, &task->mutex));

  // Copy the result
  *result = task->result;

  // Unlock the task mutex
  PTHREAD_ERROR_CHECK(pthread_mutex_unlock(&task->mutex));

  // Free the task and arguments
  free(task->args);
  free(task);

  return 0;
}
