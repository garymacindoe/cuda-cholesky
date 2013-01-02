#include "cutaskqueue.h"
#include <stdlib.h>
#include <assert.h>

static CUresult function() { return CUDA_SUCCESS; }

int main() {
  CUtaskqueue queue;
  CUtask task0, task1, task2, task;

  assert(cuTaskCreate(&task0, function, NULL, 0) == CUDA_SUCCESS);
  assert(cuTaskCreate(&task1, function, NULL, 0) == CUDA_SUCCESS);
  assert(cuTaskCreate(&task2, function, NULL, 0) == CUDA_SUCCESS);

  assert(cuTaskQueueCreate(&queue, 2) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 0);
  assert(cuTaskQueueIsEmpty(queue));
  assert(cuTaskQueuePeek(queue, &task) == CUDA_ERROR_INVALID_VALUE);
  assert(cuTaskQueuePop(queue, &task) == CUDA_ERROR_INVALID_VALUE);

  assert(cuTaskQueuePush(queue, task0) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePeek(queue, &task) == CUDA_SUCCESS);
  assert(task == task0);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  cuTaskQueueClear(queue);

  assert(cuTaskQueueSize(queue) == 0);
  assert(cuTaskQueueIsEmpty(queue));
  assert(cuTaskQueuePeek(queue, &task) == CUDA_ERROR_INVALID_VALUE);
  assert(cuTaskQueuePop(queue, &task) == CUDA_ERROR_INVALID_VALUE);

  assert(cuTaskQueuePush(queue, task0) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePeek(queue, &task) == CUDA_SUCCESS);
  assert(task == task0);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePush(queue, task1) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 2);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePeek(queue, &task) == CUDA_SUCCESS);
  assert(task == task0);

  assert(cuTaskQueueSize(queue) == 2);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePop(queue, &task) == CUDA_SUCCESS);
  assert(task == task0);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePush(queue, task1) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 2);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePush(queue, task2) == CUDA_SUCCESS);

  assert(cuTaskQueueSize(queue) == 3);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePop(queue, &task) == CUDA_SUCCESS);
  assert(task == task1);

  assert(cuTaskQueueSize(queue) == 2);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePop(queue, &task) == CUDA_SUCCESS);
  assert(task == task1);

  assert(cuTaskQueueSize(queue) == 1);
  assert(!cuTaskQueueIsEmpty(queue));

  assert(cuTaskQueuePop(queue, &task) == CUDA_SUCCESS);
  assert(task == task2);

  assert(cuTaskQueueSize(queue) == 0);
  assert(cuTaskQueueIsEmpty(queue));

  cuTaskQueueDestroy(queue);

  assert(cuTaskExecute(task0) == CUDA_SUCCESS);
  assert(cuTaskExecute(task1) == CUDA_SUCCESS);
  assert(cuTaskExecute(task2) == CUDA_SUCCESS);

  assert(cuTaskDestroy(task0, NULL) == CUDA_SUCCESS);
  assert(cuTaskDestroy(task1, NULL) == CUDA_SUCCESS);
  assert(cuTaskDestroy(task2, NULL) == CUDA_SUCCESS);

  return CUDA_SUCCESS;
}
