size_t FUNCTION(block,width) (const TYPE(block) * b) { return b->width; }
size_t FUNCTION(block,height) (const TYPE(block) * b) { return b->height; }
size_t FUNCTION(block,pitch) (const TYPE(block) * b) { return b->pitch; }
CUdeviceptr FUNCTION(block,data) (const TYPE(block) * b) { return b->data; }
CUcontext FUNCTION(block,context) (const TYPE(block) * b) { return b->context; }
