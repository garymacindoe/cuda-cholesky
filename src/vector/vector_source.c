int FUNCTION(vector, Create)(TYPE(  vector) * v, int n) {
  if ((v->data = malloc(n * sizeof(SCALAR))) == NULL)   // Aligned to 16 byte boundary by glibc.
    return ENOMEM;
  v->n = n;
  v->inc = 1;
  return 0;
}

CUresult FUNCTION(cuVector, Create)(TYPE(CUvector) * v, int n) {
  if (n > 0)
    CU_ERROR_CHECK(cuMemAlloc(&v->data, n * sizeof(SCALAR)));       // Aligned to 256 byte boundary.
  else
    v->data = 0;
  v->n = n;
  v->inc = 1;
  return CUDA_SUCCESS;
}

void FUNCTION(vector, Destroy)(TYPE(  vector) * v) {
  free(v->ptr);
}

CUresult FUNCTION(cuVector, Destroy)(TYPE(CUvector) * v) {
  if (v->data != 0)
    CU_ERROR_CHECK(cuMemFree(v->data));
  return CUDA_SUCCESS;
}


void FUNCTION(vector, Copy)(TYPE(vector) * dst, const TYPE(vector) * src) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    memcpy(dst->data, src->data, dst->n * sizeof(SCALAR));
  else {        // Else do each element individually
    for (int i = 0; i < dst->n; i++)
      dst->data[i * dst->inc] = src->data[i * src->inc];
  }
}

CUresult FUNCTION(cuVector, Copy)(TYPE(CUvector) * dst, const TYPE(CUvector) * src) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyDtoD(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else {        // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = src->data;
    p.srcPitch = (unsigned int)(src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = dst->data;
    p.dstPitch = (unsigned int)(dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = (unsigned int)sizeof(SCALAR);
    p.Height = (unsigned int)dst->n;

    // CUDA Toolkit Reference Manual, version 4.0 (February 2011), section 4.34, page 233:
    //  "On intra-device memory copies (device to device, CUDA array to device, CUDA array to CUDA array),
    //   cuMemcpy2D() may fail for pitches not computed by cuMemAllocPitch(). cuMemcpy2DUnaligned() does not
    //   have this restriction, but may run significantly slower in the cases where cuMemcpy2D() would have
    //   returned an error code."
    if (p.dstPitch % 256 == 0 && p.srcPitch % 256 == 0)
      CU_ERROR_CHECK(cuMemcpy2D(&p));
    else
      CU_ERROR_CHECK(cuMemcpy2DUnaligned(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuVector, CopyAsync)(TYPE(CUvector) * dst, const TYPE(CUvector) * src, CUstream stream) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyDtoD(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else {        // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = src->data;
    p.srcPitch = (unsigned int)(src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = dst->data;
    p.dstPitch = (unsigned int)(dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = (unsigned int)sizeof(SCALAR);
    p.Height = (unsigned int)dst->n;

    // CUDA Toolkit Reference Manual, version 4.0 (February 2011), section 4.34, page 233:
    //  "On intra-device memory copies (device to device, CUDA array to device, CUDA array to CUDA array),
    //   cuMemcpy2D() may fail for pitches not computed by cuMemAllocPitch(). cuMemcpy2DUnaligned() does not
    //   have this restriction, but may run significantly slower in the cases where cuMemcpy2D() would have
    //   returned an error code."
    if (p.dstPitch % 256 == 0 && p.srcPitch % 256 == 0)
      CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
    else
      CU_ERROR_CHECK(cuMemcpy2DUnaligned(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuVector, CopyHtoD)(TYPE(CUvector) * dst, const TYPE(vector) * src) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyHtoD(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else {  // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost = src->data;
    p.srcPitch = (unsigned int)(src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = dst->data;
    p.dstPitch = (unsigned int)(dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = (unsigned int)sizeof(SCALAR);
    p.Height = (unsigned int)dst->n;

    CU_ERROR_CHECK(cuMemcpy2D(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuVector, CopyHtoDAsync)(TYPE(CUvector) * dst, const TYPE(vector) * src, CUstream stream) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyHtoD(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else if (src->pinned) {  // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost = src->data;
    p.srcPitch = (unsigned int)(src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = dst->data;
    p.dstPitch = (unsigned int)(dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = (unsigned int)sizeof(SCALAR);
    p.Height = (unsigned int)dst->n;

    CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
  }
  else {
    for (int i = 0; i < dst->n; i++)
      CU_ERROR_CHECK(cuMemcpyHtoDAsync(dst->data + i * dst->inc * sizeof(SCALAR), &src->data[i * src->inc], sizeof(SCALAR), stream));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuVector, CopyDtoH)(TYPE(vector) * dst, const TYPE(CUvector) * src) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyDtoH(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else {  // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = src->data;
    p.srcPitch = (unsigned int)(src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.dstHost = dst->data;
    p.dstPitch = (unsigned int)(dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = (unsigned int)sizeof(SCALAR);
    p.Height = (unsigned int)dst->n;

    CU_ERROR_CHECK(cuMemcpy2D(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuVector, CopyDtoHAsync)(TYPE(vector) * dst, const TYPE(CUvector) * src, CUstream stream) {
  // If the elements are contiguous then we can do a fast copy
  if (dst->inc == 1 && src->inc == 1)
    CU_ERROR_CHECK(cuMemcpyDtoH(dst->data, src->data, dst->n * sizeof(SCALAR)));
  else {  // Do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = src->data;
    p.srcPitch = (src->inc * sizeof(SCALAR));
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.dstHost = dst->data;
    p.dstPitch = (dst->inc * sizeof(SCALAR));
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = sizeof(SCALAR);
    p.Height = dst->n;

    CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
  }
  return CUDA_SUCCESS;
}


size_t FUNCTION(vector, FRead )(TYPE(vector) * v, FILE * file) {
  if (v->inc == 1)
    return fread(v->data, sizeof(SCALAR), (size_t)v->n, file);

  size_t res = 0, r;
  for (int i = 0; i < v->n; i++, res += r) {
    if ((r = fread(&v->data[i * v->inc], sizeof(SCALAR), 1, file)) != 1)
      return res;
  }
  return res;
}

size_t FUNCTION(vector, FWrite)(const TYPE(vector) * v, FILE * file) {
  if (v->inc == 1)
    return fwrite(v->data, sizeof(SCALAR), (size_t)v->n, file);

  size_t res = 0, r;
  for (int i = 0; i < v->n; i++, res += r) {
    if ((r = fwrite(&v->data[i * v->inc], sizeof(SCALAR), 1, file)) != 1)
      return res;
  }
  return res;
}

int FUNCTION(vector, FScanF)(FILE * file, TYPE(vector) * v) {
  int res = 0, r;

  for (int i = 0; i < v->n; i++, res += r) {
#ifdef COMPLEX
    BASE_TYPE real, imag;
    if ((r = fscanf(file, "%" SCN, &real)) != 1)
      return (r < 0) ? r : res;
    if ((r = fscanf(file, "%" SCN, &imag)) != 1)
      return (r < 0) ? r : res;
    v->data[i * v->inc] = real + I * imag;
#else
    if ((r = fscanf(file, "%" SCN, &v->data[i * v->inc])) != 1)
      return (r < 0) ? r : res;
#endif
  }

  return res;
}

int FUNCTION(vector, FPrintF)(FILE * file, const char * fmt, const TYPE(vector) * v) {
  int res = 0, r;

  for (int i = 0; i < v->n; i++, res += r) {
#ifdef COMPLEX
    if ((r = fprintf(file, fmt, creal(v->data[i * v->inc]), cimag(v->data[i * v->inc]))) < 0)
#else
    if ((r = fprintf(file, fmt, v->data[i * v->inc])) < 0)
#endif
      return r;
  }

  return res;
}


void FUNCTION(vector, SetAll)(TYPE(vector) * v, SCALAR a) {
  const int n = v->n;
  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
      v->data[i] = a;
  }
  else {
    for (int i = 0; i < n; i++)
      v->data[i * v->inc] = a;
  }
}

CUresult FUNCTION(cuVector, SetAll)(TYPE(CUvector) * v, SCALAR a, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z3allI" MANGLE "EvPT_mS0_m"));

  void * params[] = { &v->data, &v->inc, &a, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(vector, SetBasis)(TYPE(vector) * v, int index) {
  const int n = v->n;
  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
#ifdef COMPLEX
      v->data[i] = LITERAL(0, 0);
    v->data[index] = LITERAL(1, 0);
#else
      v->data[i] = LITERAL(0);
    v->data[index] = LITERAL(1);
#endif
  }
  else {
    for (int i = 0; i < n; i++)
#ifdef COMPLEX
      v->data[i * v->inc] = LITERAL(0, 0);
    v->data[index * v->inc] = LITERAL(1, 0);
#else
      v->data[i * v->inc] = LITERAL(0);
    v->data[index * v->inc] = LITERAL(1);
#endif
  }
}

CUresult FUNCTION(cuVector, SetBasis)(TYPE(CUvector) * v, int index, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z5basisI" MANGLE "EvPT_mmm"));

  void * params[] = { &v->data, &v->inc, &index, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, Add)(TYPE(  vector) * x, const TYPE(  vector) * y) {
  const int n = x->n;
  SCALAR * xdata = x->data;
  const SCALAR * ydata = y->data;

  // If the elements are contiguous then we can use vectorisation
  if (x->inc == 1 && y->inc == 1) {
    for (int i = 0; i < n; i++)
      xdata[i] += ydata[i];
  }
  else {
    for (int i = 0; i < n; i++)
      xdata[i * x->inc] += ydata[i * y->inc];
  }
}

CUresult FUNCTION(cuVector, Add)(TYPE(CUvector) * x, const TYPE(CUvector) * y, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z3addI" MANGLE "EvPT_mS1_mm"));

  void * params[] = { &x->data, &x->inc, (void *)&y->data, (void *)&y->inc, &x->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)x->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, AddConst)(TYPE(  vector) * v, SCALAR a) {
  const int n = v->n;
  SCALAR * data = v->data;

  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
      data[i] += a;
  }
  else {
    for (int i = 0; i < n; i++)
      data[i * v->inc] += a;
  }
}

CUresult FUNCTION(cuVector, AddConst)(TYPE(CUvector) * v, SCALAR a, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8addConstI" MANGLE "EvPT_mS0_m"));

  void * params[] = { &v->data, &v->inc, &a, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, Subtract)(TYPE(  vector) * x, const TYPE(  vector) * y) {
  const int n = x->n;
  SCALAR * xdata = x->data;
  const SCALAR * ydata = y->data;

  // If the elements are contiguous then we can use vectorisation
  if (x->inc == 1 && y->inc == 1) {
    for (int i = 0; i < n; i++)
      xdata[i] -= ydata[i];
  }
  else {
    for (int i = 0; i < n; i++)
      xdata[i * x->inc] -= ydata[i * y->inc];
  }
}

CUresult FUNCTION(cuVector, Subtract)(TYPE(CUvector) * x, const TYPE(CUvector) * y, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8subtractI" MANGLE "EvPT_mS1_mm"));

  void * params[] = { &x->data, &x->inc, (void *)&y->data, (void *)&y->inc, &x->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)x->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, SubtractConst)(TYPE(  vector) * v, SCALAR a) {
  const int n = v->n;
  SCALAR * data = v->data;

  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
      data[i] -= a;
  }
  else {
    for (int i = 0; i < n; i++)
      data[i * v->inc] -= a;
  }
}

CUresult FUNCTION(cuVector, SubtractConst)(TYPE(CUvector) * v, SCALAR a, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z13subtractConstI" MANGLE "EvPT_mS0_m"));

  void * params[] = { &v->data, &v->inc, &a, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, Multiply)(TYPE(  vector) * x, const TYPE(  vector) * y) {
  const int n = x->n;
  SCALAR * xdata = x->data;
  const SCALAR * ydata = y->data;

  // If the elements are contiguous then we can use vectorisation
  if (x->inc == 1 && y->inc == 1) {
    for (int i = 0; i < n; i++)
      xdata[i] *= ydata[i];
  }
  else {
    for (int i = 0; i < n; i++)
      xdata[i * x->inc] *= ydata[i * y->inc];
  }
}

CUresult FUNCTION(cuVector, Multiply)(TYPE(CUvector) * x, const TYPE(CUvector) * y, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8multiplyI" MANGLE "EvPT_mS1_mm"));

  void * params[] = { &x->data, &x->inc, (void *)&y->data, (void *)&y->inc, &x->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)x->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, MultiplyConst)(TYPE(  vector) * v, SCALAR a) {
  const int n = v->n;
  SCALAR * data = v->data;

  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
      data[i] *= a;
  }
  else {
    for (int i = 0; i < n; i++)
      data[i * v->inc] *= a;
  }
}

CUresult FUNCTION(cuVector, MultiplyConst)(TYPE(CUvector) * v, SCALAR a, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z13multiplyConstI" MANGLE "EvPT_mS0_m"));

  void * params[] = { &v->data, &v->inc, &a, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, Divide)(TYPE(  vector) * x, const TYPE(  vector) * y) {
  const int n = x->n;
  SCALAR * xdata = x->data;
  const SCALAR * ydata = y->data;

  // If the elements are contiguous then we can use vectorisation
  if (x->inc == 1 && y->inc == 1) {
    for (int i = 0; i < n; i++)
      xdata[i] /= ydata[i];
  }
  else {
    for (int i = 0; i < n; i++)
      xdata[i * x->inc] /= ydata[i * y->inc];
  }
}

CUresult FUNCTION(cuVector, Divide)(TYPE(CUvector) * x, const TYPE(CUvector) * y, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z6divideI" MANGLE "EvPT_mS1_mm"));

  void * params[] = { &x->data, &x->inc, (void *)&y->data, (void *)&y->inc, &x->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)x->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void     FUNCTION(  vector, DivideConst)(TYPE(  vector) * v, SCALAR a) {
  const int n = v->n;
  SCALAR * data = v->data;

  if (v->inc == 1) {
    for (int i = 0; i < n; i++)
      data[i] /= a;
  }
  else {
    for (int i = 0; i < n; i++)
      data[i * v->inc] /= a;
  }
}

CUresult FUNCTION(cuVector, DivideConst)(TYPE(CUvector) * v, SCALAR a, CUstream stream) {
  CUmodule module;
  CU_ERROR_CHECK(cuMultiGPULoadModule(&module, "vector" STRING(SHORT)));

  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z11divideConstI" MANGLE "EvPT_mS0_m"));

  void * params[] = { &v->data, &v->inc, &a, &v->n };

  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)v->n + 511) / 512), 1, 1, 512, 1, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}
