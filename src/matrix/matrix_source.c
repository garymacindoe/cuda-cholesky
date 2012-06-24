CUresult FUNCTION(matrix, Create)(TYPE(matrix) * A, size_t m, size_t n) {
  // Round column length in bytes (m * sizeof(SCALAR)) up to memory alignment
  A->ld = align(m * sizeof(SCALAR), memory_alignment);

  // Allocate n columns of ld bytes + extra space to align pointer
  MALLOC_CHECK(A->ptr = (SCALAR *)malloc(n * A->ld + memory_alignment));

  A->m = m;
  A->n = n;
  A->ld /= sizeof(SCALAR);      // ld is expressed as a number of elements
  A->data = (SCALAR *)align((size_t)A->ptr, memory_alignment);       // Align pointer (not needed with GLibC)
  A->pinned = false;

  return CUDA_SUCCESS;
}

CUresult FUNCTION(matrix, CreatePinned)(TYPE(matrix) * A, size_t m, size_t n) {
  // Round column length in bytes (m * sizeof(SCALAR)) up to memory alignment
  A->ld = align(m * sizeof(SCALAR), memory_alignment);

  // Allocate n columns of ld bytes + extra space to align pointer
  CU_ERROR_CHECK(cuMemAllocHost((void **)&A->ptr, n * A->ld + memory_alignment));

  A->m = m;
  A->n = n;
  A->ld /= sizeof(SCALAR);      // ld is expressed as a number of elements
  A->data = (SCALAR *)align((size_t)A->ptr, memory_alignment);       // Align pointer (not needed with GLibC)
  A->pinned = true;

  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, Create)(TYPE(CUmatrix) * A, size_t m, size_t n) {
  // cuMemAllocPitch automatically rounds up column length in bytes to GPU memory alignment (256 bytes)
  // cuMemAllocPitch will return an error if asked to allocate 0 bytes
  if (m > 0 && n > 0)
    CU_ERROR_CHECK(cuMemAllocPitch(&A->data, &A->ld, m * sizeof(SCALAR), n, sizeof(SCALAR)));
  else {
    A->data = 0;
    A->ld = 0;
  }

  A->m = m;
  A->n = n;
  A->ld /= sizeof(SCALAR);      // ld is expressed as a number of elements for CUBLAS

  return CUDA_SUCCESS;
}

CUresult FUNCTION(matrix, Destroy)(TYPE(matrix) * A) {
  if (A->pinned)
    CU_ERROR_CHECK(cuMemFreeHost(A->ptr));
  else
    free(A->ptr);
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, Destroy)(TYPE(CUmatrix) * A) {
  if (A->data != 0)       // cuMemFree returns an error if asked to free a NULL pointer
    CU_ERROR_CHECK(cuMemFree(A->data));
  return CUDA_SUCCESS;
}


void FUNCTION(matrix, Copy)(TYPE(matrix) * A, const TYPE(matrix) * B) {
  // If the elements are contiguous do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    memcpy(A->data, B->data, A->m * A->n * sizeof(SCALAR));
  else {        // Else copy each column individually
    for (size_t j = 0; j < A->n; j++)
      memcpy(&A->data[j * A->ld], &B->data[j * B->ld], A->m * sizeof(SCALAR));
  }
}

CUresult FUNCTION(cuMatrix, Copy)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B) {
  // If the elements are contiguous do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyDtoD(A->data, B->data, A->m * A->n * sizeof(SCALAR)));
  else {
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2D(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, CopyAsync)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // If the elements are contiguous do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyDtoDAsync(A->data, B->data, A->m * A->n * sizeof(SCALAR), stream));
  else {
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, CopyHtoD)(TYPE(CUmatrix) * A, const TYPE(matrix) * B) {
  // If the elements are contiguous then do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyHtoD(A->data, B->data, A->m * A->n * sizeof(SCALAR)));
  else {
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2D(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, CopyHtoDAsync)(TYPE(CUmatrix) * A, const TYPE(matrix) * B, CUstream stream) {
  // If the elements are contiguous then do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyHtoDAsync(A->data, B->data, A->m * A->n * sizeof(SCALAR), stream));
  else if (B->pinned) { // Asynchronous 2D memcpys only work with pinned host memory
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_HOST;
    p.srcHost = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    p.dstDevice = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
  }
  else {
    for (size_t j = 0; j < A->n; j++)
      CU_ERROR_CHECK(cuMemcpyHtoDAsync(A->data + j * A->ld * sizeof(SCALAR), &B->data[j * B->ld], A->m * sizeof(SCALAR), stream));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, CopyDtoH)(TYPE(matrix) * A, const TYPE(CUmatrix) * B) {
  // If the elements are contiguous then do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyDtoH(A->data, B->data, A->m * A->n * sizeof(SCALAR)));
  else {
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.dstHost = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2D(&p));
  }
  return CUDA_SUCCESS;
}

CUresult FUNCTION(cuMatrix, CopyDtoHAsync)(TYPE(matrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // If the elements are contiguous then do a fast copy
  if (A->ld == A->m && B->ld == B->m)
    CU_ERROR_CHECK(cuMemcpyDtoHAsync(A->data, B->data, A->m * A->n * sizeof(SCALAR), stream));
  else if (A->pinned) { // Asynchronous 2D memcpys only work with pinned host memory
    // Get the GPU to do a 2D memcpy
    CUDA_MEMCPY2D p;

    p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    p.srcDevice = B->data;
    p.srcPitch = B->ld * sizeof(SCALAR);
    p.srcXInBytes = 0;
    p.srcY = 0;

    p.dstMemoryType = CU_MEMORYTYPE_HOST;
    p.dstHost = A->data;
    p.dstPitch = A->ld * sizeof(SCALAR);
    p.dstXInBytes = 0;
    p.dstY = 0;

    p.WidthInBytes = A->m * sizeof(SCALAR);
    p.Height = A->n;

    CU_ERROR_CHECK(cuMemcpy2DAsync(&p, stream));
  }
  else {
    for (size_t j = 0; j < A->n; j++)
      CU_ERROR_CHECK(cuMemcpyDtoHAsync(&A->data[j * A->ld], B->data + j * B->ld * sizeof(SCALAR), A->m * sizeof(SCALAR), stream));
  }
  return CUDA_SUCCESS;
}


size_t FUNCTION(matrix, FRead)(TYPE(matrix) * A, FILE * file) {
  // If the elements are contiguous read the whole matrix at once
  if (A->m == A->ld)
    return fread(A->data, sizeof(SCALAR), A->m * A->n, file);

  // Read each column individually
  size_t res = 0, r;
  for (size_t j = 0; j < A->n; j++, res += r) {
    if ((r = fread(&A->data[j * A->ld], sizeof(SCALAR), A->m, file)) < A->m)
      return res + r;
  }
  return res;
}

size_t FUNCTION(matrix, FWrite)(const TYPE(matrix) * A, FILE * file) {
  // If the elements are contiguous write the whole matrix at once
  if (A->m == A->ld)
    return fwrite(A->data, sizeof(SCALAR), A->m * A->n, file);

  // Write each column individually
  size_t res = 0, r;
  for (size_t j = 0; j < A->n; j++, res += r) {
    if ((r = fwrite(&A->data[j * A->ld], sizeof(SCALAR), A->m, file)) < A->m)
      return res + r;
  }
  return res;
}

int FUNCTION(matrix, FScanF)(FILE * file, TYPE(matrix) * A) {
  // The number of items successfully matched
  int res = 0, r;

  // Humans read matrices in row-major order
  for (size_t i = 0; i < A->m; i++) {
    for (size_t j = 0; j < A->n; j++, res += r) {
#ifdef COMPLEX
      BASE_TYPE real, imag;
      if ((r = fscanf(file, "%" SCN, &real)) != 1)
        return (r < 0) ? r : res;
      if ((r = fscanf(file, "%" SCN, &imag)) != 1)
        return (r < 0) ? r : res;
      A->data[j * A->ld + i] = real + I * imag;
#else
      if ((r = fscanf(file, "%" SCN, &A->data[j * A->ld + i])) != 1)
        return (r < 0) ? r : res;
#endif
    }
  }

  return res;   // Will be m*n for a successful read
}

int FUNCTION(matrix, FPrintF)(FILE * file, const char * fmt, const TYPE(matrix) * A) {
  // The number of characters successfully written
  int res = 0, r;

  // Humans read matrices in row-major order
  for (size_t i = 0; i < A->m; i++) {
    for (size_t j = 0; j < A->n; j++, res += r) {
      // If a negative value was returned there was an error
      if ((r = fprintf(file, fmt, A->data[j * A->ld + i])) < 0)
        return r;
    }
    // fputc returns EOF on error
    if (fputc('\n', file) == EOF) return EOF;
    // Increment the number of characters printed to include the newline
    res++;
  }

  return res;   // Depends on output format and size of matrix
}


void FUNCTION(matrix, SetAll)(TYPE(matrix) * A, SCALAR alpha) {
  // For GCC's automatic vectorisation to work it needs to know that m and ld cannot be aliased
  const size_t m = A->m;
  const size_t ld = A->ld;
  for (size_t j = 0; j < A->n; j++)
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      A->data[j * ld + i] = alpha;
}

CUresult FUNCTION(cuMatrix, SetAll)(TYPE(CUmatrix) * A, SCALAR alpha, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z3allI" MANGLE "EvPT_mS0_mm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &alpha, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, SetIdentity)(TYPE(matrix) * A) {
  // For GCC's automatic vectorisation to work it needs to know that m and ld cannot be aliased
  const size_t m = A->m;
  const size_t ld = A->ld;

  // Zero the matrix using vectorised code
  for (size_t j = 0; j < A->n; j++) {
    SCALAR * column = &A->data[j * ld]; // Prevent aliasing of column
    for (size_t i = 0; i < m; i++)      // Vectorised loop
#ifdef COMPLEX
      column[i] = LITERAL(0, 0);
#else
      column[i] = LITERAL(0);
#endif
  }

  // Set the diagonal elements to 1
  const size_t n = (A->m < A->n) ? A->m : A->n;     // Prevent aliasing of n
  for (size_t i = 0; i < n; i++)        // Vectorised loop (if ld is small)
#ifdef COMPLEX
    A->data[i * ld + i] = LITERAL(1, 0);
#else
    A->data[i * ld + i] = LITERAL(1);
#endif
}

CUresult FUNCTION(cuMatrix, SetIdentity)(TYPE(CUmatrix) * A, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8identityI" MANGLE "EvPT_mmm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, Add)(TYPE(matrix) * A, const TYPE(matrix) * B) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * columnA = &A->data[j * A->ld];     // Prevent aliasing of columns
    SCALAR * columnB = &B->data[j * A->ld];
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      columnA[i] += columnB[i];
  }
}

CUresult FUNCTION(cuMatrix, Add)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z3addI" MANGLE "EvPT_mS1_mmm"));

  // Set up the parameters (need to cast away const nature of B)
  void * params[] = { &A->data, &A->ld, (void *)&B->data, (void *)&B->ld, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, AddConst)(TYPE(matrix) * A, SCALAR alpha) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * column = &A->data[j * A->ld];      // Prevent aliasing of column
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      column[i] += alpha;
  }
}

CUresult FUNCTION(cuMatrix, AddConst)(TYPE(CUmatrix) * A, SCALAR alpha, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8addConstI" MANGLE "EvPT_mS0_mm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &alpha, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, Subtract)(TYPE(matrix) * A, const TYPE(matrix) * B) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * columnA = &A->data[j * A->ld];     // Prevent aliasing of columns
    SCALAR * columnB = &B->data[j * A->ld];
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      columnA[i] -= columnB[i];
  }
}

CUresult FUNCTION(cuMatrix, Subtract)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8subtractI" MANGLE "EvPT_mS1_mmm"));

  // Set up the parameters (need to cast away const nature of B)
  void * params[] = { &A->data, &A->ld, (void *)&B->data, (void *)&B->ld, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, SubtractConst)(TYPE(matrix) * A, SCALAR alpha) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * column = &A->data[j * A->ld];      // Prevent aliasing of column
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      column[i] -= alpha;
  }
}

CUresult FUNCTION(cuMatrix, SubtractConst)(TYPE(CUmatrix) * A, SCALAR alpha, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z13subtractConstI" MANGLE "EvPT_mS0_mm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &alpha, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, Multiply)(TYPE(matrix) * A, const TYPE(matrix) * B) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * columnA = &A->data[j * A->ld];     // Prevent aliasing of columns
    SCALAR * columnB = &B->data[j * A->ld];
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      columnA[i] *= columnB[i];
  }
}

CUresult FUNCTION(cuMatrix, Multiply)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z8multiplyI" MANGLE "EvPT_mS1_mmm"));

  // Set up the parameters (need to cast away const nature of B)
  void * params[] = { &A->data, &A->ld, (void *)&B->data, (void *)&B->ld, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, MultiplyConst)(TYPE(matrix) * A, SCALAR alpha) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * column = &A->data[j * A->ld];      // Prevent aliasing of column
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      column[i] *= alpha;
  }
}

CUresult FUNCTION(cuMatrix, MultiplyConst)(TYPE(CUmatrix) * A, SCALAR alpha, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z13multiplyConstI" MANGLE "EvPT_mS0_mm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &alpha, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, Divide)(TYPE(matrix) * A, const TYPE(matrix) * B) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * columnA = &A->data[j * A->ld];     // Prevent aliasing of columns
    SCALAR * columnB = &B->data[j * A->ld];
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      columnA[i] /= columnB[i];
  }
}

CUresult FUNCTION(cuMatrix, Divide)(TYPE(CUmatrix) * A, const TYPE(CUmatrix) * B, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z6divideI" MANGLE "EvPT_mS1_mmm"));

  // Set up the parameters (need to cast away const nature of B)
  void * params[] = { &A->data, &A->ld, (void *)&B->data, (void *)&B->ld, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}


void FUNCTION(matrix, DivideConst)(TYPE(matrix) * A, SCALAR alpha) {
  // For GCC's automatic vectorisation to work it needs to know that m cannot be aliased
  const size_t m = A->m;

  for (size_t j = 0; j < A->n; j++) {
    SCALAR * column = &A->data[j * A->ld];      // Prevent aliasing of column
    for (size_t i = 0; i < m; i++)      // Vectorised loop
      column[i] /= alpha;
  }
}

CUresult FUNCTION(cuMatrix, DivideConst)(TYPE(CUmatrix) * A, SCALAR alpha, CUstream stream) {
  // Get a pointer to the singleton module
  CUmodule module;
  CU_ERROR_CHECK(cuModuleLoadResource(&module, "matrix" STRING(SHORT)));

  // Load the function from the module
  CUfunction function;
  CU_ERROR_CHECK(cuModuleGetFunction(&function, module, "_Z11divideConstI" MANGLE "EvPT_mS0_mm"));

  // Set up the parameters
  void * params[] = { &A->data, &A->ld, &alpha, &A->m, &A->n };

  // Launch the function on the GPU
  CU_ERROR_CHECK(cuLaunchKernel(function, max(1, ((unsigned int)A->m + 31) / 32), max(1, ((unsigned int)A->n + 15) / 16), 1, 32, 16, 1, 0, stream, params, NULL));

  return CUDA_SUCCESS;
}
