#ifndef MULTIGPU_H
#define MULTIGPU_H

#include "cumultigpu.h"
#include "task.h"

/**
 * Runs a task using a particular CUDA context.
 *
 * @param mGPU  the multiGPU context.
 * @param i     the index of the CUDA context to use.
 * @param task  the task to run.
 * @return CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY,
 *         CUDA_ERROR_OPERATING_SYSTEM.
 */
CUresult cuMultiGPURunTask(CUmultiGPU, int, CUtask);

#endif
