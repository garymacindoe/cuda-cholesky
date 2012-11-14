#ifndef __HANDLE_H
#define __HANDLE_H

#include "cuhandle.h"

#define CU_HANDLE_GEMM 0
#define CU_HANDLE_SYRK 1
#define CU_HANDLE_HERK 2
#define CU_HANDLE_TRSM 3
#define CU_HANDLE_TRMM 4

#define CU_HANDLE_POTRF 5
#define CU_HANDLE_POTRI 6
#define CU_HANDLE_TRTRI 7

#define CU_HANDLE_SINGLE         0
#define CU_HANDLE_DOUBLE         1
#define CU_HANDLE_COMPLEX        2
#define CU_HANDLE_DOUBLE_COMPLEX 3

/**
 * Returns the context associated with the handle in which all cached resources
 * are allocated.
 */
CUcontext cuHandleGetContext(CUhandle);

/**
 * Loads a module for a linear algebra function into the context and caches it
 * for future use.
 *
 * @param handle     the handle to load the module into.
 * @param module     the module is returned through this pointer.
 * @param precision  the precision of the module to load (CU_HANDLE_SINGLE,
 *                   CU_HANDLE_DOUBLE, CU_HANDLE_COMPLEX or
 *                   CU_HANDLE_DOUBLE_COMPLEX).
 * @param function   the type independent name of the function to load.
 * @return CUDA_ERROR_INVALID_VALUE if the precision or function name are
 *         invalid or any error associated with cuModuleLoad.
 */
CUresult cuHandleGetModule(CUhandle, CUmodule *, int, int);

/* Pre-allocated linear and 2D memory */
CUresult cuHandleMemAlloc(CUhandle, CUdeviceptr *, size_t);
CUresult cuHandleMemAllocHost(CUhandle, void **, size_t);
CUresult cuHandleMemAllocPitch(CUhandle, CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
CUresult cuHandleMemAllocPitchHost(CUhandle, void **, size_t *, size_t, size_t, unsigned int);

/* Lazy allocated streams per context */
CUresult cuHandleGetStream(CUhandle, CUstream *, int);

#endif
