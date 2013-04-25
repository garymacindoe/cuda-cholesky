#ifndef HANDLE_H
#define HANDLE_H

struct __culapackhandle_st {
  CUBLAShandle blas_handle;
  CUcontext context;
  CUmodule spotrf, strtri, slauum, slogdet;
  CUmodule dpotrf, dtrtri, dlauum, dlogdet;
};

struct __cumultigpulapackhandle_st {
  CUmultiGPUBLAShandle blas_handle;
};

#endif
