#ifndef HANDLE_H
#define HANDLE_H

struct __culapackhandle_st {
  CUBLAShandle blas_handle;
  CUcontext context;
  CUmodule spotrf, strtri, slauum;
  CUmodule cpotrf, ctrtri, clauum;
  CUmodule dpotrf, dtrtri, dlauum;
  CUmodule zpotrf, ztrtri, zlauum;
  CUmodule slogdet, clogdet, dlogdet, zlogdet;
};

struct __cumultigpulapackhandle_st {
  CUmultiGPUBLAShandle blas_handle;
};

#endif
