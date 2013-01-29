#include "cumultigpu.h"
#include <assert.h>

struct args_st {
  const int x, y;
};

CUresult add(const void * args) {
  struct args_st * a = (struct args_st *)args;
  return a->x + a->y;
}

CUresult success(const void * args) {
  (void)args;
  return 0;
}

int main() {
  struct args_st args = { 2, 5 };

  CUtask task;
  CUresult result;

  /* Creating a task with a NULL function results in CUDA_ERROR_INVALID_VALUE */
  assert(cuTaskCreate(&task, NULL, &args, sizeof(struct args_st)) == CUDA_ERROR_INVALID_VALUE);
  /* Creating a task with NULL arguments and a non-zero size results int CUDA_ERROR_INVALID_VALUE */
  assert(cuTaskCreate(&task, add, NULL, sizeof(struct args_st)) == CUDA_ERROR_INVALID_VALUE);

  /* Creating a task with a petabyte of arguments results in CUDA_ERROR_OUT_OF_MEMORY */
  assert(cuTaskCreate(&task, add, &args, 1099511627776ul) == CUDA_ERROR_OUT_OF_MEMORY);

  /* Creating a task with a non-NULL function, non-NULL args and non-zero size results in CUDA_SUCCESS */
  assert(cuTaskCreate(&task, add, &args, sizeof(struct args_st)) == CUDA_SUCCESS);

  /* Executing a task results in CUDA_SUCCESS */
  assert(cuTaskExecute(task) == CUDA_SUCCESS);

  /* Destroying the task and getting the result works */
  assert(cuTaskDestroy(task, &result) == CUDA_SUCCESS);

  /* The answer is correct */
  assert(result == 7);

  /* Creating a task with NULL arguments works if size is zero */
  assert(cuTaskCreate(&task, success, NULL, 0) == CUDA_SUCCESS);

  /* Executing a task results in CUDA_SUCCESS */
  assert(cuTaskExecute(task) == CUDA_SUCCESS);

  /* Destroying the task and getting the result works */
  assert(cuTaskDestroy(task, &result) == CUDA_SUCCESS);

  /* The answer is correct */
  assert(result == 0);

  return 0;
}
