#ifndef ERROR_H
#define ERROR_H

#include <stdbool.h>
#include <string.h>
#include <cuda.h>

// Expand and stringify argument
#define STRINGx(x) #x
#define STRING(x) STRINGx(x)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error handler function type.
 *
 * @param call      function call that threw the error.
 * @param function  calling function where the error occurred.
 * @param file      file the error occurred in.
 * @param line      line number the error occurred on.
 * @param error     integer error code.
 * @param strerror  error description function.
 */
typedef void (*errorHandler_t)(const char *, const char *, const char *, int,
                               int, const char * (*)(int));

/**
 * CUDA Driver API error handler (may be NULL).
 */
extern errorHandler_t errorHandler;

/**
 * CUDA Driver API error string function.
 *
 * @param error  the CUDA driver API error code.
 * @return a human-readable description of the error that occurred.
 */
const char * cuGetErrorString(CUresult);

#define CU_ERROR_CHECK(call) \
  do { \
    CUresult __error__; \
    if ((__error__ = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))cuGetErrorString); \
      return __error__; \
    } \
  } while (false)

#define ERROR_CHECK(call) \
  do { \
    int __error__; \
    if ((__error__ = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))strerror); \
      return CUDA_ERROR_OPERATING_SYSTEM; \
    } \
  } while (false)

#define ERROR_CHECK_VOID(call) \
  do { \
    int __error__; \
    if ((__error__ = (call)) != CUDA_SUCCESS) { \
      if (errorHandler != NULL) \
        errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, \
                     (const char * (*)(int))strerror); \
      return; \
    } \
  } while (false)

#ifdef __cplusplus
}
#endif

#endif
