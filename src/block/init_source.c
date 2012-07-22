TYPE(block) * FUNCTION(block,alloc)(const size_t width, const size_t height) {
  if (n == 0)
    GSL_ERROR_VAL("block length n must be positive integer", GSL_EINVAL, NULL);

  TYPE(block) * b = malloc(sizeof(TYPE(block)));
  if (b == NULL)
    GSL_ERROR_VAL("failed to allocate space for block struct", GSL_ENOMEM, NULL);

  CUresult error = cuCtxGetCurrent(&b->context);
  if (error != CUDA_SUCCESS) {
    free(b);
    GSL_ERROR_VAL("failed to get current GPU context for block", error + 64, NULL);
  }

  error = cuMemAllocPitch(&b->data, &b->pitch, width * sizeof(SCALAR), height, sizeof(SCALAR));
  if (error != CUDA_SUCCESS) {
    free(b);
    GSL_ERROR_VAL("failed to allocate space for block data", error + 64, NULL);
  }

  b->width = width;
  b->height = height;

  return b;
}

TYPE(block) * FUNCTION(block,calloc)(const size_t width, const size_t height) {
  TYPE(block) * b = FUNCTION(block,alloc)(width, height);

  if (b == NULL)
    return NULL;



  return b;
}

void FUNCTION(block,free)(TYPE(block) * b) {
  if (b == NULL)
    return;

  CUresult error = cuMemFree(b->data);
  if (error != CUDA_SUCCESS)
    GSL_ERROR_VOID("failed to free space for block data", error + 64, NULL);

  free(b);
}
