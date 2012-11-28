#ifndef ERROR_H
#define ERROR_H

#include <stdbool.h>
#include <cuda.h>
#include <cublas_v2.h>

// Expand and stringify argument
#define STRINGx(x) #x
#define STRING(x) STRINGx(x)

/**
 * Function to transfer integer error codes into character string descriptions.
 *
 * @param error  integer error code.
 * @return human-readable description of the error code.
 */
typedef const char * (*strerror_t)(int);

/**
 * Error handler function type.
 *
 * @param call      the function call that threw the error.
 * @param function  the calling function where the error occurred.
 * @param file      the file the error occured in.
 * @param line      the line number within the file.
 * @param error     the integer error code.
 * @param strerror  the error description function for the type of error that
 *                  occurred.
 */
typedef void (*errorHandler_t)(const char *, const char *, const char *,
                               unsigned int, int, strerror_t);

/**
 * Error handler (may be NULL).
 */
extern errorHandler_t errorHandler;

/**
 * CUDA Driver API error string function.
 *
 * @param error  the CUDA driver API error code.
 * @return a human-readable description of the error that occurred.
 */
const char * cuGetErrorString(CUresult);

/**
 * CUBLAS error string function.
 *
 * @param error  the CUBLAS error status.
 * @return a human-readable description on the error that occurred.
 */
const char * cublasGetErrorString(cublasStatus_t);

#define ERROR_CHECK(call) \
 do { \
   int __error__; \
   if ((__error__ = (call)) != 0) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, strerror); \
     return __error__; \
   } \
 } while (false)

#define ERROR_CHECK_VOID(call) \
 do { \
   int __error__; \
   if ((__error__ = (call)) != 0) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, strerror); \
     return; \
   } \
 } while (false)

#define ERROR_CHECK_VAL(call, val) \
 do { \
   int __error__; \
   if ((__error__ = (call)) != 0) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, strerror); \
     return val; \
   } \
 } while (false)

#define MALLOC_CHECK(allocation) \
 do { \
   if ((allocation) == NULL) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(allocation), __func__, __FILE__, __LINE__, CUDA_ERROR_OUT_OF_MEMORY, (strerror_t)cuGetErrorString); \
     return CUDA_ERROR_OUT_OF_MEMORY; \
   } \
 } while (false)

#define MALLOC_CHECK_VOID(allocation) \
 do { \
   if ((allocation) == NULL) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(allocation), __func__, __FILE__, __LINE__, CUDA_ERROR_OUT_OF_MEMORY, (strerror_t)cuGetErrorString); \
     return; \
   } \
 } while (false)

#define MALLOC_CHECK_VAL(allocation, val) \
 do { \
   if ((allocation) == NULL) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(allocation), __func__, __FILE__, __LINE__, CUDA_ERROR_OUT_OF_MEMORY, (strerror_t)cuGetErrorString); \
     return val; \
   } \
 } while (false)

#define CU_ERROR_CHECK(call) \
 do { \
   CUresult __error__; \
   if ((__error__ = (call)) != CUDA_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, (strerror_t)cuGetErrorString); \
     return __error__; \
   } \
 } while (false)

#define CU_ERROR_CHECK_VOID(call) \
 do { \
   CUresult __error__; \
   if ((__error__ = (call)) != CUDA_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, (strerror_t)cuGetErrorString); \
   return; \
   } \
 } while (false)

#define CU_ERROR_CHECK_VAL(call, val) \
 do { \
   CUresult __error__; \
   if ((__error__ = (call)) != CUDA_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __error__, (strerror_t)cuGetErrorString); \
     return val; \
   } \
 } while (false)

#define CUBLAS_ERROR_CHECK(call) \
 do { \
   cublasStatus_t __status__; \
   if ((__status__ = (call)) != CUBLAS_STATUS_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __status__, (strerror_t)cublasGetErrorString); \
     return __status__; \
   } \
 } while (false)

#define CUBLAS_ERROR_CHECK_VOID(call) \
 do { \
   cublasStatus_t __status__; \
   if ((__status__ = (call)) != CUBLAS_STATUS_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __status__, (strerror_t)cublasGetErrorString); \
     return; \
   } \
 } while (false)

#define CUBLAS_ERROR_CHECK_VAL(call, val) \
 do { \
   cublasStatus_t __status__; \
   if ((__status__ = (call)) != CUBLAS_STATUS_SUCCESS) { \
     if (errorHandler != NULL) \
       errorHandler(STRING(call), __func__, __FILE__, __LINE__, __status__, (strerror_t)cublasGetErrorString); \
     return val; \
   } \
 } while (false)

#endif
