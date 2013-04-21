#ifndef HANDLE_H
#define HANDLE_H

struct __culapackhandle_st {
  CUBLAShandle blas_handle;
  CUcontext context;
  CUmodule spotrf, spotfimm2, strtri, strtimm2, slauum, slogdet;
};

struct __cumultigpulapackhandle_st {
  CUmultiGPUBLAShandle blas_handle;
};

#endif
